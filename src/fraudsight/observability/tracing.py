"""Tracing: JSONL by default, Langfuse and OTel when configured.

The default exporter writes newline-delimited JSON to `artifacts/traces/`. That
is not a compromise -- it is the format that makes a trace *greppable*, which is
what you actually want at 2am when one triage produced a strange brief and you
need the six spans that led to it. A hosted UI is better for aggregate
questions; a file is better for "show me this transaction".

Span names follow the OpenTelemetry GenAI semantic conventions where they exist
(`gen_ai.*` attributes on model calls) so that pointing an OTel collector at
this system does not require re-instrumenting it. Vendor-specific span shapes
are the reason observability migrations take a quarter.

Every span carries the trace id AND the transaction id. Linking to a session id
alone is the common mistake: fraud triage has no session, and the question being
asked is always about a transaction.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fraudsight.config import Settings, settings as default_settings
from fraudsight.triage.state import TriageState

# OTel GenAI semantic-convention attribute names, spelled once.
GEN_AI = {
    "system": "gen_ai.system",
    "model": "gen_ai.request.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "operation": "gen_ai.operation.name",
}


class Exporter(Protocol):
    def export(self, trace: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


@dataclass
class JsonlExporter:
    path: Path

    def export(self, trace: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, default=str) + "\n")

    def flush(self) -> None:
        return None


@dataclass
class LangfuseExporter:  # pragma: no cover - requires a running Langfuse
    """Maps our span tree onto Langfuse's trace/generation model."""

    public_key: str
    secret_key: str
    host: str
    _client: Any = None

    def __post_init__(self) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=self.public_key, secret_key=self.secret_key, host=self.host
        )

    def export(self, trace: dict[str, Any]) -> None:
        root = self._client.trace(
            id=trace.get("trace_id"),
            name="triage.run",
            metadata={
                "transaction_id": trace.get("transaction_id"),
                "band": trace.get("band"),
                "outcome": trace.get("outcome"),
                "degraded": trace.get("degraded"),
            },
            tags=[trace.get("band", "unknown"), trace.get("outcome", "unknown")],
        )
        for span in trace.get("spans", []):
            name = span.get("name", "span")
            if name.startswith("llm."):
                root.generation(
                    name=name,
                    model=trace.get("model"),
                    usage={
                        "input": trace.get("input_tokens", 0),
                        "output": trace.get("output_tokens", 0),
                        "unit": "TOKENS",
                    },
                    metadata=span.get("attributes", {}),
                )
            else:
                root.span(
                    name=name,
                    metadata=span.get("attributes", {}),
                    level="ERROR" if span.get("error") else "DEFAULT",
                )

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


@dataclass
class OTelExporter:  # pragma: no cover - requires a collector
    """OTLP export using GenAI semantic conventions."""

    endpoint: str
    _tracer: Any = None

    def __post_init__(self) -> None:
        from opentelemetry import trace as ot
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=self.endpoint)))
        ot.set_tracer_provider(provider)
        self._tracer = ot.get_tracer("fraudsight")

    def export(self, trace: dict[str, Any]) -> None:
        with self._tracer.start_as_current_span("triage.run") as root:
            root.set_attribute("fraudsight.transaction_id", trace.get("transaction_id", ""))
            root.set_attribute(GEN_AI["system"], trace.get("provider") or "none")
            root.set_attribute(GEN_AI["model"], trace.get("model") or "none")
            root.set_attribute(GEN_AI["input_tokens"], trace.get("input_tokens", 0))
            root.set_attribute(GEN_AI["output_tokens"], trace.get("output_tokens", 0))
            for span in trace.get("spans", []):
                with self._tracer.start_as_current_span(span["name"]) as child:
                    for k, v in (span.get("attributes") or {}).items():
                        child.set_attribute(f"fraudsight.{k}", str(v))


@dataclass
class Tracer:
    """Fan-out to every configured exporter. Never raises into the caller.

    An observability backend that takes the request path down with it is worse
    than no observability, so export failures are counted and swallowed. The
    count is on `/health`, which is where you notice.
    """

    exporters: list[Exporter] = field(default_factory=list)
    export_failures: int = 0
    exported: int = 0

    @classmethod
    def build(cls, cfg: Settings | None = None) -> Tracer:
        cfg = cfg or default_settings
        cfg.ensure_dirs()
        exporters: list[Exporter] = [JsonlExporter(cfg.trace_dir / "triage.jsonl")]

        if cfg.langfuse_public_key and cfg.langfuse_secret_key:
            try:
                exporters.append(
                    LangfuseExporter(
                        cfg.langfuse_public_key.get_secret_value(),
                        cfg.langfuse_secret_key.get_secret_value(),
                        cfg.langfuse_host,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                exporters.append(OTelExporter(endpoint))
            except Exception:  # noqa: BLE001
                pass

        return cls(exporters=exporters)

    def record(self, state: TriageState, *, trace_id: str | None = None) -> dict[str, Any]:
        trace = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **state.trace(),
        }
        for exporter in self.exporters:
            try:
                exporter.export(trace)
            except Exception:  # noqa: BLE001 - never break the request path
                self.export_failures += 1
        self.exported += 1
        return trace

    def flush(self) -> None:
        for exporter in self.exporters:
            try:
                exporter.flush()
            except Exception:  # noqa: BLE001
                self.export_failures += 1

    def health(self) -> dict[str, Any]:
        return {
            "exporters": [type(e).__name__ for e in self.exporters],
            "exported": self.exported,
            "export_failures": self.export_failures,
        }


@contextmanager
def span(state: TriageState, name: str, **attributes: Any) -> Iterator[None]:
    """Time a block and attach it to the state's span list.

    Records the span even when the body raises -- a node that failed after 4
    seconds is the span you most want to see, and a `finally` that skips it on
    the error path is how latency investigations end in silence.
    """
    import time

    from fraudsight.triage.state import Span

    t0 = time.perf_counter()
    error: str | None = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.spans.append(
            Span(
                name=name,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                attributes=attributes,
                error=error,
            )
        )


def read_traces(
    path: Path | None = None,
    limit: int | None = None,
    day: str | None = None,
) -> list[dict[str, Any]]:
    """Read traces, optionally restricted to one UTC day.

    The file is append-only across every run the repository has ever done, so a
    caller that wants "yesterday" must say so. Aggregating the whole history and
    calling it a daily report attributes every historical failure to today --
    which is how the first version of the report announced a 59% policy-block
    rate on a run that had none.
    """
    p = path or (default_settings.trace_dir / "triage.jsonl")
    if not p.exists():
        return []
    rows = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if day:
        rows = [r for r in rows if str(r.get("ts", "")).startswith(day)]
    return rows[-limit:] if limit else rows
