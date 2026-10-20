"""The contract between the triage agent and everything downstream.

This schema is the reason the LLM output is safe to build on. Nothing consumes
free text: the analyst console renders these fields, the auto-routing logic
branches on `recommendation` and `confidence`, and the eval suite scores against
them. If the model cannot produce this shape, the repair loop runs; if repair
fails, the case escalates to a human. There is no path where prose gets parsed
with a regex.

`missing_information` is the field that matters most and that portfolio agents
almost always omit. An agent that reports what it could not retrieve lets the
router escalate on incomplete evidence rather than on low confidence in a
confident-sounding hallucination.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Recommendation(StrEnum):
    APPROVE = "approve"
    BLOCK = "block"
    ESCALATE = "escalate_to_analyst"
    REQUEST_STEP_UP = "request_step_up_auth"


class Direction(StrEnum):
    RISK_INCREASING = "risk_increasing"
    RISK_DECREASING = "risk_decreasing"


class Evidence(BaseModel):
    """One cited signal. Every field exists to make the claim checkable."""

    model_config = ConfigDict(extra="forbid")

    signal: str = Field(
        description="Short snake_case name of the signal, e.g. 'velocity_spike', 'device_unseen'."
    )
    detail: str = Field(
        description=(
            "One sentence stating the observed value and why it matters. Must be traceable "
            "to a tool output -- do not state anything not present in the provided evidence."
        )
    )
    direction: Direction = Field(description="Whether this signal raises or lowers fraud risk.")
    source_tool: str = Field(
        description=(
            "Which tool produced this: customer_history, device_intel, merchant_profile, "
            "similar_cases, or transaction (the transaction record itself)."
        )
    )
    weight: float = Field(
        ge=0.0, le=1.0, description="How much this signal drove the recommendation."
    )

    @field_validator("signal")
    @classmethod
    def _snake(cls, v: str) -> str:
        cleaned = v.strip().lower().replace(" ", "_").replace("-", "_")
        if not cleaned:
            raise ValueError("signal must not be empty")
        return cleaned

    @field_validator("detail")
    @classmethod
    def _detail_substantive(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 15:
            raise ValueError("detail must be a substantive sentence, not a label")
        return v


ALLOWED_TOOLS = frozenset(
    {"customer_history", "device_intel", "merchant_profile", "similar_cases", "transaction"}
)


class TriageBrief(BaseModel):
    """What the agent must produce. Read by an analyst in ~15 seconds."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    narrative: str = Field(
        description="3-5 sentences an analyst can read in 15 seconds. No preamble, no hedging."
    )
    evidence: list[Evidence] = Field(
        min_length=2,
        max_length=8,
        description="At least two signals. Include risk-decreasing ones -- a brief that only "
        "lists incriminating evidence is not an investigation.",
    )
    similar_cases: list[str] = Field(
        description="case_id values of comparable historical cases from similar_cases. "
        "Empty list if retrieval returned nothing relevant."
    )
    recommendation: Recommendation
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated confidence in the recommendation. Low confidence must be "
        "paired with escalate_to_analyst.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Evidence you tried to obtain and could not. Drives escalation.",
    )
    agrees_with_model_score: bool = Field(
        description="Whether your assessment agrees with the statistical risk score you were "
        "given. Disagreement is useful signal, not an error -- say so plainly."
    )

    @field_validator("narrative")
    @classmethod
    def _narrative_bounds(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 80:
            raise ValueError("narrative too short to be a useful brief (min 80 chars)")
        if len(v) > 900:
            raise ValueError(f"narrative must be at most 900 chars, got {len(v)}")
        return v

    @field_validator("evidence")
    @classmethod
    def _tools_known(cls, v: list[Evidence]) -> list[Evidence]:
        bad = sorted({e.source_tool for e in v} - ALLOWED_TOOLS)
        if bad:
            raise ValueError(
                f"unknown source_tool {bad}; must be one of {sorted(ALLOWED_TOOLS)}"
            )
        return v

    def has_mitigating_evidence(self) -> bool:
        return any(e.direction is Direction.RISK_DECREASING for e in self.evidence)

    def top_signals(self, n: int = 3) -> list[Evidence]:
        return sorted(self.evidence, key=lambda e: -e.weight)[:n]


class PolicyViolation(BaseModel):
    """A brief that validates structurally can still be unsafe to act on."""

    rule: str
    detail: str
    severity: str = Field(pattern="^(warn|block)$")
