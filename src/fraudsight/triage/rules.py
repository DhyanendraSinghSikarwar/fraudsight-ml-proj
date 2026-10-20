"""The rule-based `TriageBrief` synthesiser.

Registered against the deterministic provider, this is what produces a brief
when every paid provider is unavailable, and what the eval suite scores as the
*floor* that the LLM has to beat to justify its cost. Publishing an agent's
score without a floor is publishing a number with no denominator.

The rules are deliberately conservative. A rule engine cannot weigh an unusual
combination of signals, so it escalates whenever the evidence is not lopsided:
it will not block a customer on a weighted sum. That produces a predictable
profile in the eval report -- strong on schema compliance and groundedness
(every sentence is templated from a retrieved number), weak on the ambiguous
slice (it escalates almost all of it), and that shape is the argument for having
an LLM in the loop at all.

Every sentence it emits is assembled from `Signal.detail` strings, which are
themselves built from retrieved values. Its hallucination rate is structurally
zero, which is a useful control: when the LLM's measured hallucination rate is
compared against it, the comparison isolates the model rather than the harness.
"""

from __future__ import annotations

from typing import Any

from fraudsight.llm import deterministic
from fraudsight.triage.evidence import LIFT_RISK_INCREASING
from fraudsight.triage.schema import Recommendation

# Weighted-sum thresholds. Chosen so the engine escalates by default: the
# blueprint's cost matrix prices a wrong block at $25 and an analyst review at
# $4.20, so when a rule engine is unsure, paying for the review is the cheaper
# error.
BLOCK_NET_RISK = 2.2
APPROVE_NET_RISK = -0.6
STEP_UP_NET_RISK = 1.2

MAX_EVIDENCE = 6


@deterministic.register("TriageBrief")
def synthesise_brief(evidence: dict[str, Any], system: str, user: str) -> dict[str, Any]:
    signals: list[dict[str, Any]] = list(evidence.get("derived_signals") or [])
    txn = evidence.get("transaction") or {}
    model = evidence.get("model_assessment") or {}
    score = _f(model.get("score"))
    transaction_id = str(txn.get("transaction_id") or "unknown")

    if not signals:
        # No signals at all means the evidence layer found nothing to say. The
        # honest output is an escalation that says exactly that, not a fabricated
        # two-item evidence list to satisfy the schema minimum.
        return _no_evidence_brief(transaction_id, score, evidence)

    increasing = [s for s in signals if s.get("direction") == "risk_increasing"]
    decreasing = [s for s in signals if s.get("direction") == "risk_decreasing"]
    net = sum(_f(s.get("weight")) for s in increasing) - sum(
        _f(s.get("weight")) for s in decreasing
    )

    chosen = _select_evidence(increasing, decreasing)
    recommendation, confidence = _decide(net, score, chosen, evidence)

    return {
        "transaction_id": transaction_id,
        "narrative": _narrative(chosen, net, score, recommendation, evidence),
        "evidence": [
            {
                "signal": s["signal"],
                "detail": s["detail"],
                "direction": s["direction"],
                "source_tool": s["source_tool"],
                "weight": round(_f(s.get("weight")), 3),
            }
            for s in chosen
        ],
        "similar_cases": [
            str(c.get("case_id")) for c in (evidence.get("similar_cases") or [])[:3]
        ],
        "recommendation": str(recommendation),
        "confidence": confidence,
        "missing_information": list(evidence.get("unavailable_evidence") or []),
        "agrees_with_model_score": model_says_risky(model) == (
            recommendation in (Recommendation.BLOCK, Recommendation.REQUEST_STEP_UP)
        ),
    }


def model_says_risky(model: dict[str, Any]) -> bool:
    """Is the scorer flagging this, judged against the portfolio base rate?

    Not `score >= 0.5`. At a 3.4% base rate a calibrated 0.31 means "nine times
    more likely to be fraud than a random transaction", which is the scorer
    saying yes. Comparing to a coin flip files it as exculpatory and inverts
    `agrees_with_model_score` on most of the triage band -- the band exists
    precisely because these scores are elevated but not conclusive.

    Shared with guardrails/output.py so the brief and the policy check cannot
    disagree about what the model said.
    """
    lift = _f(model.get("score_vs_base_rate"))
    if lift:
        return lift >= LIFT_RISK_INCREASING
    base = _f(model.get("portfolio_fraud_rate")) or 0.034
    return _f(model.get("score")) / base >= LIFT_RISK_INCREASING


def _select_evidence(
    increasing: list[dict[str, Any]], decreasing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Strongest signals, but always at least one from each side when both
    exist. A brief that lists only incriminating evidence fails output policy,
    and rightly -- it is not an investigation."""
    out: list[dict[str, Any]] = []
    if increasing:
        out.append(increasing[0])
    if decreasing:
        out.append(decreasing[0])
    rest = sorted(
        [s for s in [*increasing, *decreasing] if s not in out],
        key=lambda s: -_f(s.get("weight")),
    )
    out.extend(rest[: MAX_EVIDENCE - len(out)])
    if len(out) < 2:  # schema floor
        out = [*increasing, *decreasing][:2]
    return out


def _decide(
    net: float, score: float, chosen: list[dict[str, Any]], evidence: dict[str, Any]
) -> tuple[Recommendation, float]:
    """Map the weighted sum onto an action, then onto a defensible confidence.

    Confidence is derived from the *margin* past the threshold, capped well
    below what output policy requires for a unilateral block. A rule engine
    should not be 0.9-confident about anything, and pretending otherwise would
    let the floor pass policy checks the LLM has to earn.
    """
    gaps = len(evidence.get("unavailable_evidence") or [])

    if net >= BLOCK_NET_RISK and score >= 0.5 and gaps == 0:
        confidence = min(0.88, 0.80 + 0.02 * (net - BLOCK_NET_RISK))
        return Recommendation.BLOCK, round(confidence, 3)
    if net >= STEP_UP_NET_RISK:
        return Recommendation.REQUEST_STEP_UP, round(min(0.72, 0.55 + 0.05 * net), 3)
    if net <= APPROVE_NET_RISK and score < 0.3 and gaps == 0:
        return Recommendation.APPROVE, round(min(0.85, 0.80 + 0.03 * abs(net)), 3)
    # Everything else is what a rule engine genuinely cannot resolve.
    margin = abs(net)
    return Recommendation.ESCALATE, round(max(0.40, min(0.70, 0.45 + 0.08 * margin)), 3)


def _narrative(
    chosen: list[dict[str, Any]],
    net: float,
    score: float,
    recommendation: Recommendation,
    evidence: dict[str, Any],
) -> str:
    lead = {
        Recommendation.BLOCK: "The evidence points consistently to fraud.",
        Recommendation.APPROVE: "The evidence is consistent with legitimate activity.",
        Recommendation.REQUEST_STEP_UP: (
            "The evidence is mixed but leans risky; a step-up verification would resolve it."
        ),
        Recommendation.ESCALATE: "The evidence does not resolve this case either way.",
    }[recommendation]

    top = [s for s in chosen if s.get("direction") == "risk_increasing"][:2]
    mitigating = [s for s in chosen if s.get("direction") == "risk_decreasing"][:1]

    parts = [lead]
    if top:
        parts.append(" ".join(s["detail"] for s in top))
    if mitigating:
        parts.append("Against that: " + mitigating[0]["detail"])
    parts.append(
        f"The statistical scorer put this at {score:.3f} and the weighted signal balance is "
        f"{net:+.2f}."
    )
    gaps = evidence.get("unavailable_evidence") or []
    if gaps:
        parts.append(f"{len(gaps)} piece(s) of evidence could not be retrieved.")

    text = " ".join(p.strip() for p in parts if p).strip()
    if len(text) > 900:
        text = text[:880].rsplit(" ", 1)[0] + " ..."
    if len(text) < 80:  # schema floor; only reachable with one terse signal
        text += " Rule-based triage could not assemble a fuller picture from the evidence."
    return text


def _no_evidence_brief(transaction_id: str, score: float, evidence: dict[str, Any]) -> dict[str, Any]:
    detail = (
        f"The statistical scorer returned {score:.3f} but no contextual evidence was "
        f"retrievable for this transaction."
    )
    return {
        "transaction_id": transaction_id,
        "narrative": (
            "No contextual evidence could be assembled for this transaction, so no "
            "investigation is possible from the automated layer. "
            + detail
            + " Routing to an analyst with the raw authorisation."
        ),
        "evidence": [
            {
                "signal": "model_risk_score",
                "detail": detail,
                "direction": "risk_increasing" if score >= 0.5 else "risk_decreasing",
                "source_tool": "transaction",
                "weight": round(min(0.9, 0.3 + score), 3),
            },
            {
                "signal": "no_context_available",
                "detail": (
                    "Every context tool returned empty for this transaction, so the score "
                    "cannot be corroborated or contradicted."
                ),
                "direction": "risk_increasing",
                "source_tool": "transaction",
                "weight": 0.4,
            },
        ],
        "similar_cases": [],
        "recommendation": str(Recommendation.ESCALATE),
        "confidence": 0.4,
        "missing_information": list(evidence.get("unavailable_evidence") or []) or [
            "all context tools returned empty"
        ],
        "agrees_with_model_score": score >= 0.5,
    }


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
