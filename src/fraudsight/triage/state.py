"""Typed agent state.

The state object is the whole memory of a triage run. Everything a node needs is
on it, nodes return a new state rather than mutating shared objects, and the
final state is what gets serialised into the trace. That has two consequences
worth the discipline:

* **A run is replayable.** `TriageState.model_dump()` at any node boundary is
  enough to resume from that boundary, which is what makes a failed run
  debuggable rather than a mystery.
* **Nothing is implicit.** There is no scratchpad dict, no `context` bag, no
  attribute added at runtime. If a node produces something, it is a field here,
  and the type checker knows about it.

`usage` accumulates across repairs rather than being overwritten. A brief that
needed two repair attempts cost three model calls, and a cost dashboard that
records only the last one under-reports exactly the cases that are expensive.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fraudsight.llm.base import Usage
from fraudsight.triage.schema import TriageBrief


class Outcome(StrEnum):
    COMPLETED = "completed"
    ESCALATED_VALIDATION = "escalated_validation_failed"
    ESCALATED_POLICY = "escalated_policy_blocked"
    ESCALATED_ERROR = "escalated_provider_error"
    ESCALATED_BUDGET = "escalated_budget_exhausted"


class Span(BaseModel):
    """One node's execution, in the shape the trace exporter expects."""

    name: str
    duration_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TriageState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- inputs ---------------------------------------------------------
    transaction_id: str
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    score: float = 0.0
    band: str = "triage"

    # -- gathered evidence ----------------------------------------------
    bundle: Any = None  # EvidenceBundle; not a pydantic model by design
    signals: list[dict[str, Any]] = Field(default_factory=list)
    untrusted: dict[str, Any] = Field(default_factory=dict)

    # -- guardrail results ----------------------------------------------
    redactions: dict[str, int] = Field(default_factory=dict)
    injection: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)

    # -- model output ---------------------------------------------------
    brief: TriageBrief | None = None
    validation_error: str | None = None
    repairs: int = 0
    # The assembled user turn, kept so the repair node can re-send the identical
    # evidence with only the rejection appended. Rebuilding it would risk
    # repairing against a different bundle than the one that failed.
    user_turn: str = Field(default="", repr=False)

    # -- accounting -----------------------------------------------------
    usage: Usage | None = None
    spans: list[Span] = Field(default_factory=list)
    route: dict[str, Any] = Field(default_factory=dict)
    prompt_ids: dict[str, str] = Field(default_factory=dict)

    # -- result ---------------------------------------------------------
    outcome: Outcome | None = None
    error: str | None = None

    def add_usage(self, usage: Usage) -> None:
        self.usage = usage if self.usage is None else self.usage.merge(usage)

    def add_span(self, name: str, duration_ms: float, **attributes: Any) -> None:
        self.spans.append(
            Span(name=name, duration_ms=round(duration_ms, 3), attributes=attributes)
        )

    @property
    def total_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 3)

    @property
    def cost_usd(self) -> float:
        return round(self.usage.cost_usd, 8) if self.usage else 0.0

    @property
    def degraded(self) -> bool:
        return bool(self.usage and self.usage.degraded)

    def trace(self) -> dict[str, Any]:
        """The row written to artifacts/traces/. Deliberately flat -- a trace
        nobody can grep is a trace nobody reads."""
        return {
            "transaction_id": self.transaction_id,
            "outcome": str(self.outcome) if self.outcome else None,
            "score": round(self.score, 6),
            "band": self.band,
            "recommendation": str(self.brief.recommendation) if self.brief else None,
            "confidence": self.brief.confidence if self.brief else None,
            "evidence_count": len(self.brief.evidence) if self.brief else 0,
            "missing_information": list(self.brief.missing_information) if self.brief else [],
            "repairs": self.repairs,
            "total_ms": self.total_ms,
            "cost_usd": self.cost_usd,
            "degraded": self.degraded,
            "provider": self.usage.provider if self.usage else None,
            "model": self.usage.model if self.usage else None,
            "input_tokens": self.usage.input_tokens if self.usage else 0,
            "output_tokens": self.usage.output_tokens if self.usage else 0,
            "cache_read_tokens": self.usage.cache_read_tokens if self.usage else 0,
            "injection_detected": bool(self.injection.get("detected")),
            "injection_categories": self.injection.get("categories", []),
            "redactions": self.redactions,
            "policy_violations": [
                v.get("rule") for v in self.policy.get("violations", [])
            ],
            "prompt_ids": self.prompt_ids,
            "route": self.route,
            "spans": [s.model_dump() for s in self.spans],
            "error": self.error,
        }
