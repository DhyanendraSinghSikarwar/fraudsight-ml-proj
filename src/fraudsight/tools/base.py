"""Tool contract, registry, and the read-only scoping that makes tools safe.

The tool-execution boundary is the real security perimeter of an LLM system, so
three rules are enforced here rather than trusted to prompt wording:

* **Every tool is read-only.** There is no write path in the registry. An agent
  cannot block a card, refund a charge, or email a customer -- those actions are
  taken by the analyst console after a human clicks, which is the human-approval
  gate the NFRs require for irreversible actions.

* **Every call is scoped to the transaction under triage.** `ToolRegistry.call`
  takes the scope from the transaction record, not from the model. An agent that
  asks for a different customer's history gets `ScopeViolation`, and the attempt
  is recorded -- because a prompt-injected agent asking for the wrong customer is
  precisely the signal you want in the trace rather than a quiet success.

* **Every result is provenanced.** `ToolResult.source` and `.fields` are what
  the groundedness judge checks `Evidence.detail` against. A claim the agent
  cannot trace to a `ToolResult` field is a hallucination by definition, and
  that definition is mechanical rather than a matter of taste.

`ToolRegistry.gather` runs the independent tools concurrently. That is the
single biggest latency win in the triage path and it is measured, not assumed --
see the before/after numbers in artifacts/optimisation_curve.json.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class ScopeViolation(RuntimeError):
    """A tool was asked for an entity outside the transaction under triage."""


class ToolError(RuntimeError):
    """The tool ran and could not answer. Distinct from returning 'not found',
    which is a legitimate answer and becomes `missing_information`."""


@dataclass(frozen=True)
class ToolResult:
    """What one tool returned, with everything needed to audit the claim."""

    tool: str
    found: bool
    fields: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    # Populated when the tool read attacker-controlled free text (merchant
    # names, memo fields). The prompt builder wraps these differently -- see
    # guardrails/injection.py.
    untrusted_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "found": self.found,
            "source": self.source,
            "latency_ms": self.latency_ms,
            **({"error": self.error} if self.error else {}),
            "fields": dict(self.fields),
        }

    def claim_values(self) -> set[str]:
        """Every scalar an `Evidence.detail` may legitimately cite.

        Used by the groundedness check. Numbers are rendered in several forms
        because a model writing prose says "1,240" or "1240.0" for the same
        underlying 1240.
        """
        out: set[str] = set()
        for v in _walk(self.fields):
            out.update(_renderings(v))
        return out


def _walk(node: Any) -> list[Any]:
    if isinstance(node, Mapping):
        return [x for v in node.values() for x in _walk(v)]
    if isinstance(node, (list, tuple)):
        return [x for v in node for x in _walk(v)]
    return [node]


def _renderings(v: Any) -> set[str]:
    if v is None or isinstance(v, bool):
        return {str(v).lower()}
    if isinstance(v, int):
        return {str(v), f"{v:,}"}
    if isinstance(v, float):
        return {
            f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:,.2f}", f"{v:,.0f}", str(v), str(round(v, 2))
        }
    return {str(v).strip().lower()}


class Tool(Protocol):
    name: str
    description: str

    async def __call__(self, scope: Mapping[str, Any]) -> ToolResult: ...


@dataclass
class ToolSpec:
    """A registered tool. `description` is a prompt -- it is what the model reads
    to decide whether the tool is relevant, and vague descriptions are the
    documented number-one cause of wrong tool calls."""

    name: str
    description: str
    fn: Callable[[Mapping[str, Any]], Awaitable[ToolResult]]
    scope_keys: tuple[str, ...]
    reads_untrusted_text: bool = False


class ToolRegistry:
    def __init__(self, *, max_concurrency: int = 8, timeout_s: float = 2.0) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s
        self.scope_violations: list[dict[str, Any]] = []

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "description": s.description, "scope": ", ".join(s.scope_keys)}
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    async def call(self, name: str, scope: Mapping[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            # Allow-list, not a deny-list: an unknown name is refused rather
            # than attempted.
            raise ScopeViolation(f"tool {name!r} is not in the allow-list {self.names}")

        missing = [k for k in spec.scope_keys if scope.get(k) in (None, "")]
        if missing:
            self.scope_violations.append({"tool": name, "missing_scope": missing})
            raise ScopeViolation(
                f"{name} requires scope keys {missing} from the transaction under triage"
            )

        t0 = time.perf_counter()
        try:
            async with self._sem:
                result = await asyncio.wait_for(spec.fn(scope), timeout=self._timeout_s)
        except TimeoutError:
            return ToolResult(
                tool=name, found=False, latency_ms=_ms(t0),
                error=f"timed out after {self._timeout_s}s",
            )
        except ToolError as exc:
            return ToolResult(tool=name, found=False, latency_ms=_ms(t0), error=str(exc))
        return result

    async def gather(self, names: list[str], scope: Mapping[str, Any]) -> dict[str, ToolResult]:
        """Run tools concurrently. A failure in one does not lose the others --
        a partial evidence bundle plus an honest `missing_information` entry is
        a better brief than no brief."""

        async def one(n: str) -> tuple[str, ToolResult]:
            try:
                return n, await self.call(n, scope)
            except ScopeViolation as exc:
                return n, ToolResult(tool=n, found=False, error=f"scope violation: {exc}")

        pairs = await asyncio.gather(*(one(n) for n in names))
        return dict(pairs)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)
