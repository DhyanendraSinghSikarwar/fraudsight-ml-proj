"""The triage state machine.

    redact -> gather -> retrieve -> reason -> validate -> (repair -> validate)* -> end

**Why this is a hand-written runner rather than a LangGraph import.** The
blueprint specifies LangGraph, and `langgraph_adapter.py` compiles these exact
nodes into a real `StateGraph` when the `agent` extra is installed -- that path
is tested and supported. But the default runner is the ~80 lines below, for
reasons the blueprint itself argues for in its escalation ladder:

* This graph is a fixed six-node DAG with one bounded loop. Everything LangGraph
  earns its weight on -- cyclic planning, dynamic routing, interrupt/resume
  across a queue, multi-agent handoff -- is absent here. Adding the dependency
  would buy state management this file does in a dataclass.
* The triage path has a 6s p95 SLA and runs on every ambiguous transaction. A
  framework import that adds even 200ms of cold start is a real cost against a
  real budget.
* `make demo` has to run on a reviewer's laptop with no services. Every
  optional dependency is a reviewer who does not run the project.

The honest condition under which this decision flips is in
docs/adr/0004-no-agent-framework.md: the moment triage needs durable
human-in-the-loop interrupts across process restarts, this runner stops being
adequate and the adapter becomes the default.

**Ordering is load-bearing.** `redact` runs first and unconditionally -- there is
no path from raw transaction to provider that skips it, which is what makes the
security test a statement about the system rather than about one code path.
`gather` runs the three independent tools concurrently; `retrieve` depends on
the signals derived from what `gather` returned, so it cannot be folded in.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraudsight.config import Settings, settings as default_settings
from fraudsight.data.profiles import ProfileTables, portfolio_baseline
from fraudsight.guardrails import injection, output, pii
from fraudsight.llm.base import BudgetExceeded, ProviderError, SchemaValidationError
from fraudsight.llm.router import LLMRouter
from fraudsight.tools.base import ToolRegistry, ToolResult
from fraudsight.tools.similar_cases import build_query
from fraudsight.triage import prompts
from fraudsight.triage.evidence import EvidenceBundle, derive_signals
from fraudsight.triage.schema import Recommendation, TriageBrief
from fraudsight.triage.state import Outcome, TriageState

Node = Callable[["TriageContext", TriageState], Awaitable[TriageState]]

GATHER_TOOLS = ["customer_history", "device_intel", "merchant_profile"]


@dataclass
class TriageContext:
    """Everything the nodes need, assembled once per process rather than per
    transaction. Loading the profile tables per triage would dominate the
    latency budget."""

    router: LLMRouter
    tools: ToolRegistry
    profiles: ProfileTables
    cfg: Settings
    portfolio: dict[str, float]
    checkpoint_dir: Path | None = None

    @classmethod
    def build(
        cls,
        router: LLMRouter,
        tools: ToolRegistry,
        profiles: ProfileTables,
        cfg: Settings | None = None,
        *,
        checkpoint_dir: Path | None = None,
    ) -> TriageContext:
        cfg = cfg or default_settings
        return cls(
            router=router,
            tools=tools,
            profiles=profiles,
            cfg=cfg,
            portfolio=portfolio_baseline(profiles),
            checkpoint_dir=checkpoint_dir,
        )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


async def redact_pii(ctx: TriageContext, state: TriageState) -> TriageState:
    """Redact before anything else touches the transaction.

    Runs on the raw record, so every downstream node -- tools, prompt builder,
    trace writer -- sees redacted values only. Redacting at the prompt boundary
    instead would leave PII in the tool results and therefore in the traces.
    """
    t0 = time.perf_counter()
    cleaned, hits = pii.redact_deep(dict(state.raw))
    state.raw = cleaned
    state.redactions = {}
    for h in hits:
        state.redactions[h.kind] = state.redactions.get(h.kind, 0) + 1
    state.add_span("guardrails.redact_pii", _ms(t0), redactions=state.redactions)
    return state


async def gather_evidence(ctx: TriageContext, state: TriageState) -> TriageState:
    """The three entity tools, concurrently.

    Sequential calls measured 3x this node's latency on the same tools. That is
    the single largest latency win in the triage path and it is reproducible via
    `fraudsight optimise --measure`.
    """
    t0 = time.perf_counter()
    scope = _scope(state.raw)
    results = await ctx.tools.gather(GATHER_TOOLS, scope)

    missing = [name for name, r in results.items() if not r.found]
    bundle = EvidenceBundle(
        transaction_id=state.transaction_id,
        transaction=_transaction_view(state.raw),
        model=_model_view(state),
        tools=results,
        velocity=state.raw.get("__velocity__", {}),
        portfolio=ctx.portfolio,
        missing=[f"{n}: {results[n].fields.get('reason', 'not found')}" for n in missing],
    )
    state.bundle = bundle
    state.add_span(
        "tools.gather",
        _ms(t0),
        tools=list(results),
        found=[n for n, r in results.items() if r.found],
        per_tool_ms={n: r.latency_ms for n, r in results.items()},
    )
    return state


async def retrieve_similar(ctx: TriageContext, state: TriageState) -> TriageState:
    """Signal-shaped retrieval. Depends on `gather`, so it cannot be parallel
    with it -- the query is built from what the entity tools returned."""
    t0 = time.perf_counter()
    bundle: EvidenceBundle = state.bundle
    signals = derive_signals(bundle)
    query = build_query([s.name for s in signals], bundle.transaction)

    result = await ctx.tools.call("similar_cases", {"query_text": query})
    bundle.tools["similar_cases"] = result
    bundle.similar_cases = list(result.fields.get("cases", []))
    if not bundle.similar_cases:
        bundle.missing.append("similar_cases: retrieval returned no comparable cases")

    # Re-derive with retrieval in hand: the similar-case base rate is itself a
    # signal, and deriving once before retrieval would omit it.
    signals = derive_signals(bundle)
    state.signals = [s.as_dict() for s in signals]
    state.add_span(
        "retrieval.similar_cases",
        _ms(t0),
        query=query,
        returned=len(bundle.similar_cases),
        mode=result.fields.get("retrieval_mode"),
    )
    return state


async def produce_brief(ctx: TriageContext, state: TriageState) -> TriageState:
    """Assemble the prompt, scan for injection, and call the model."""
    t0 = time.perf_counter()
    bundle: EvidenceBundle = state.bundle

    sanitised, scan = injection.sanitise_bundle(bundle.as_dict())
    state.injection = scan.as_dict()
    if scan.detected:
        for f in scan.findings:
            bundle.missing.append(
                f"{f.field_name}: neutralised, injection attempt ({f.category})"
            )

    state.untrusted = _untrusted_fields(sanitised)
    signals = derive_signals(bundle)
    system = prompts.load("triage_system").text
    user = prompts.build_user_turn(bundle, signals, untrusted=state.untrusted)
    state.prompt_ids = {
        "system": prompts.load("triage_system").id,
        "repair": prompts.load("triage_repair").id,
    }

    # Fail closed: the payload is checked after assembly, not before. This is the
    # last point at which a field from any of six sources can still be raw.
    pii.assert_clean(system + user, where="triage prompt")

    state.user_turn = user
    try:
        brief, usage = await ctx.router.structured(
            system=system, user=user, schema=TriageBrief, operation="triage"
        )
        state.brief = brief
        state.validation_error = None
        state.add_usage(usage)
    except SchemaValidationError as exc:
        state.validation_error = str(exc)
    except BudgetExceeded as exc:
        state.outcome = Outcome.ESCALATED_BUDGET
        state.error = str(exc)
    except ProviderError as exc:
        state.outcome = Outcome.ESCALATED_ERROR
        state.error = f"{type(exc).__name__}: {exc}"

    state.route = ctx.router.last_route.as_dict()
    state.add_span(
        "llm.produce_brief",
        _ms(t0),
        prompt_chars=len(user),
        sections=prompts.section_sizes(user),
        provider=state.usage.provider if state.usage else None,
        degraded=state.degraded,
    )
    return state


async def validate_brief(ctx: TriageContext, state: TriageState) -> TriageState:
    t0 = time.perf_counter()
    if state.brief is None:
        state.add_span("guardrails.validate_output", _ms(t0), skipped="no brief")
        return state

    report = output.validate(state.brief, state.bundle)
    state.policy = report.as_dict()
    if report.blocked:
        state.validation_error = report.repair_instruction()
    else:
        state.validation_error = None
    state.add_span(
        "guardrails.validate_output",
        _ms(t0),
        clean=report.clean,
        blocked=report.blocked,
        rules=[v.rule for v in report.violations],
    )
    return state


async def repair_brief(ctx: TriageContext, state: TriageState) -> TriageState:
    """Bounded repair. The failure is handed back verbatim.

    Retrying the identical prompt is not repair -- a model that produced an
    ungrounded number will produce it again. What changes the outcome is telling
    it which rule it broke, which is why `describe_violation` and
    `PolicyReport.repair_instruction` render field-level detail rather than a
    generic 'invalid output'.
    """
    t0 = time.perf_counter()
    state.repairs += 1
    system = prompts.load("triage_system").text
    user = prompts.build_repair_turn(state.user_turn, state.validation_error or "unspecified")
    pii.assert_clean(system + user, where="triage repair prompt")

    try:
        brief, usage = await ctx.router.structured(
            system=system, user=user, schema=TriageBrief, operation="triage_repair"
        )
        state.brief = brief
        state.validation_error = None
        state.add_usage(usage)
    except SchemaValidationError as exc:
        state.validation_error = str(exc)
    except ProviderError as exc:
        state.outcome = Outcome.ESCALATED_ERROR
        state.error = f"{type(exc).__name__}: {exc}"
    state.add_span("llm.repair_brief", _ms(t0), attempt=state.repairs)
    return state


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class TriageGraph:
    """Explicit edges, one bounded cycle, checkpoint after every node."""

    def __init__(self, ctx: TriageContext) -> None:
        self.ctx = ctx
        self.max_repairs = ctx.cfg.triage_max_repairs

    async def run(self, transaction: Mapping[str, Any], *, score: float, band: str) -> TriageState:
        state = TriageState(
            transaction_id=str(transaction.get("TransactionID", "unknown")),
            raw=dict(transaction),
            score=score,
            band=band,
        )

        for node in (redact_pii, gather_evidence, retrieve_similar, produce_brief):
            state = await node(self.ctx, state)
            self._checkpoint(state, node.__name__)
            if state.outcome is not None:
                return self._finish(state)

        state = await validate_brief(self.ctx, state)
        self._checkpoint(state, "validate_brief")

        while state.validation_error and state.repairs < self.max_repairs:
            state = await repair_brief(self.ctx, state)
            if state.outcome is not None:
                return self._finish(state)
            state = await validate_brief(self.ctx, state)
            self._checkpoint(state, f"validate_brief_{state.repairs}")

        return self._finish(state)

    def _finish(self, state: TriageState) -> TriageState:
        if state.outcome is None:
            if state.brief is None:
                state.outcome = Outcome.ESCALATED_VALIDATION
            elif state.validation_error:
                # Repairs exhausted with a policy violation outstanding. The
                # brief is kept -- an analyst reading a flawed brief plus the
                # violation list is better served than one reading nothing --
                # but the recommendation is overridden to escalation, because
                # the system must not act on output it just rejected.
                state.outcome = Outcome.ESCALATED_POLICY
                state.brief = state.brief.model_copy(
                    update={"recommendation": Recommendation.ESCALATE}
                )
            else:
                state.outcome = Outcome.COMPLETED
        return state

    def _checkpoint(self, state: TriageState, node: str) -> None:
        if self.ctx.checkpoint_dir is None:
            return
        d = self.ctx.checkpoint_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{state.transaction_id}.json").write_text(
            json.dumps({"node": node, **state.trace()}, default=str, indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


def _scope(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The only keys a tool may be called with. Everything else on the
    transaction is invisible to the tool layer by construction."""
    return {
        "card1": raw.get("card1"),
        "addr1": raw.get("addr1"),
        "DeviceInfo": raw.get("DeviceInfo"),
        "ProductCD": raw.get("ProductCD"),
        "P_emaildomain": raw.get("P_emaildomain"),
    }


def _transaction_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Analyst-readable transaction. Named fields, not the 186-column vector --
    a model handed `V258: 1.4` cannot say anything useful about it, and the
    tokens are better spent on evidence it can reason over."""
    return {
        "transaction_id": str(raw.get("TransactionID", "unknown")),
        "amount": _f(raw.get("TransactionAmt")),
        "hour_of_day": _f(raw.get("hour_of_day")),
        "day_of_week": _f(raw.get("day_of_week")),
        "product_code": raw.get("ProductCD"),
        "card_network": raw.get("card4"),
        "card_type": raw.get("card6"),
        "purchaser_email_domain": raw.get("P_emaildomain"),
        "recipient_email_domain": raw.get("R_emaildomain"),
        "device_type": raw.get("DeviceType"),
        "device_info": raw.get("DeviceInfo"),
        "billing_region": raw.get("addr1"),
        "amount_is_round": bool(_f(raw.get("amt_is_round"))),
        "email_domains_match": _f(raw.get("email_domains_match"), -1.0),
        "has_identity_record": _f(raw.get("has_identity_record")),
        "card_seen_count": _f(raw.get("card1_seen_count")),
        "device_seen_count": _f(raw.get("device_seen_count")),
        "missing_field_count": _f(raw.get("missing_field_count")),
    }


def _model_view(state: TriageState) -> dict[str, Any]:
    return {
        "score": round(state.score, 6),
        "band": state.band,
        "scorer": "LightGBM + isotonic calibration",
        "top_contributions": state.raw.get("__explanation__", []),
        "calibration_note": (
            "The score is a calibrated probability: among transactions scored near this "
            "value, approximately this fraction are fraudulent."
        ),
    }


def _untrusted_fields(bundle_dict: Mapping[str, Any]) -> dict[str, Any]:
    txn = bundle_dict.get("transaction", {}) or {}
    dev = bundle_dict.get("device_intel", {}) or {}
    merch = bundle_dict.get("merchant_profile", {}) or {}
    out = {
        "device_info": txn.get("device_info"),
        "purchaser_email_domain": txn.get("purchaser_email_domain"),
        "recipient_email_domain": txn.get("recipient_email_domain"),
        "device_key": dev.get("device_key"),
        "top_browser": dev.get("top_browser"),
        "merchant_key": merch.get("merchant_key"),
    }
    return {k: v for k, v in out.items() if v not in (None, "", "__missing__")}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def tool_results_summary(results: Mapping[str, ToolResult]) -> dict[str, Any]:
    return {n: {"found": r.found, "ms": r.latency_ms} for n, r in results.items()}
