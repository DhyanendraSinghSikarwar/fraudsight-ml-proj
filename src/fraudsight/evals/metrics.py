"""Track B metrics: how good is the triage agent?

Every metric here is defined in code rather than described in prose, because the
definition is where these numbers usually go wrong. "Agreement" in particular is
easy to inflate:

* Counting `escalate_to_analyst` as agreement with everything gives a system
  that escalates 100% of cases a perfect score. Here escalation is scored
  separately as `escalation_rate` and excluded from the agreement denominator,
  with `decisive_rate` reported alongside -- a high agreement on a 5% decisive
  rate is not the same system as a high agreement on a 60% decisive rate, and
  the pair has to be read together.
* Averaging over strata hides the only interesting part. The ambiguous stratum
  is where an LLM should beat a rule engine, so agreement is reported per
  stratum and the headline number is the macro average, not the micro -- a
  micro average would let the two easy strata carry the score.

`compare_to_floor` is what turns the numbers into a claim: the LLM's cost is
justified by the delta over the rule-based provider, not by its absolute score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

DECISIVE = {"approve", "block", "request_step_up_auth"}
ESCALATION = "escalate_to_analyst"

# How each recommendation maps onto the confirmed outcome. `request_step_up_auth`
# counts as a correct call on fraud (it stops the transaction pending
# verification) and as a *partial* miss on legitimate activity -- it adds
# friction for a genuine customer without declining them, so it is scored as
# neither agreement nor disagreement and is reported separately.
def outcome_match(recommendation: str, expected_action: str) -> bool | None:
    if recommendation == ESCALATION:
        return None
    if recommendation == "request_step_up_auth":
        return True if expected_action == "block" else None
    return recommendation == expected_action


@dataclass
class CaseResult:
    """One eval case, after running the agent over it."""

    case_id: str
    stratum: str
    label: int
    expected_action: str
    recommendation: str
    confidence: float
    schema_ok: bool
    first_attempt_ok: bool
    repairs: int
    policy_violations: list[str]
    grounded_claims: int
    total_claims: int
    evidence_count: int
    has_mitigating: bool
    missing_information: int
    latency_ms: float  # service time: sum of node spans
    wall_ms: float  # wall clock including time queued behind other cases
    cost_usd: float
    degraded: bool
    outcome: str
    tools_found: int
    similar_cases: int
    judge: dict[str, Any] | None = None

    @property
    def decisive(self) -> bool:
        return self.recommendation in DECISIVE

    @property
    def agrees(self) -> bool | None:
        return outcome_match(self.recommendation, self.expected_action)


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    if not results:
        return {"cases": 0}

    strata = sorted({r.stratum for r in results})
    per_stratum = {s: _stratum_metrics([r for r in results if r.stratum == s]) for s in strata}
    agreements = [m["agreement"] for m in per_stratum.values() if m["agreement"] is not None]

    grounded = sum(r.grounded_claims for r in results)
    claims = sum(r.total_claims for r in results)
    judged = [r for r in results if r.judge]

    return {
        "cases": len(results),
        # Macro average across strata: the ambiguous slice is a third of the set
        # and would otherwise be drowned by the two easy ones.
        "agreement_macro": round(statistics.fmean(agreements), 4) if agreements else None,
        "agreement_micro": _agreement([r for r in results]),
        "per_stratum": per_stratum,
        "decisive_rate": round(sum(r.decisive for r in results) / len(results), 4),
        "escalation_rate": round(
            sum(r.recommendation == ESCALATION for r in results) / len(results), 4
        ),
        "step_up_rate": round(
            sum(r.recommendation == "request_step_up_auth" for r in results) / len(results), 4
        ),
        "schema": {
            "pass_rate": round(sum(r.schema_ok for r in results) / len(results), 4),
            "first_attempt_pass_rate": round(
                sum(r.first_attempt_ok for r in results) / len(results), 4
            ),
            "mean_repairs": round(statistics.fmean([r.repairs for r in results]), 4),
        },
        "policy": {
            "clean_rate": round(
                sum(not r.policy_violations for r in results) / len(results), 4
            ),
            "violations_by_rule": _count([v for r in results for v in r.policy_violations]),
        },
        "groundedness": {
            "numeric_claim_grounding": round(grounded / claims, 4) if claims else 1.0,
            "claims_checked": claims,
            # The judge's rate is the semantic one; the numeric rate above is
            # mechanical. Reported separately because they measure different
            # failures and averaging them would hide both.
            "judge_hallucination_rate": (
                round(statistics.fmean([r.judge["hallucination_rate"] for r in judged]), 4)
                if judged
                else None
            ),
            "judged_cases": len(judged),
        },
        "narrative": {
            "mean_score": (
                round(statistics.fmean([r.judge["narrative_score"] for r in judged]), 3)
                if judged
                else None
            ),
            "score_histogram": _count([str(r.judge["narrative_score"]) for r in judged]),
        },
        "evidence": {
            "mean_items": round(statistics.fmean([r.evidence_count for r in results]), 2),
            "both_sides_rate": round(sum(r.has_mitigating for r in results) / len(results), 4),
            "mean_missing_information": round(
                statistics.fmean([r.missing_information for r in results]), 2
            ),
            "mean_similar_cases_cited": round(
                statistics.fmean([r.similar_cases for r in results]), 2
            ),
        },
        "confidence_calibration": confidence_buckets(results),
        # Two different questions. `latency` is what one analyst waits for;
        # `wall_clock_under_load` is what the eval harness observed with its
        # concurrency setting, and is a throughput measure. Publishing only the
        # second would overstate p95 by an order of magnitude on a CPU-bound
        # provider; publishing only the first would hide queueing entirely.
        "latency": _percentiles([r.latency_ms for r in results]),
        "wall_clock_under_load": _percentiles([r.wall_ms for r in results]),
        "cost": {
            "total_usd": round(sum(r.cost_usd for r in results), 6),
            "mean_per_triage_usd": round(
                statistics.fmean([r.cost_usd for r in results]), 8
            ),
            "per_1000_triages_usd": round(
                1000 * statistics.fmean([r.cost_usd for r in results]), 4
            ),
        },
        "degraded_rate": round(sum(r.degraded for r in results) / len(results), 4),
        "outcomes": _count([r.outcome for r in results]),
    }


def _stratum_metrics(rows: list[CaseResult]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "agreement": _agreement(rows),
        "decisive_rate": round(sum(r.decisive for r in rows) / len(rows), 4) if rows else 0.0,
        "escalation_rate": round(
            sum(r.recommendation == ESCALATION for r in rows) / len(rows), 4
        )
        if rows
        else 0.0,
        "mean_confidence": round(statistics.fmean([r.confidence for r in rows]), 4)
        if rows
        else 0.0,
        "recommendations": _count([r.recommendation for r in rows]),
    }


def _agreement(rows: list[CaseResult]) -> float | None:
    scored = [r.agrees for r in rows if r.agrees is not None]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 4)


def confidence_buckets(results: list[CaseResult], edges: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95)) -> list[dict[str, Any]]:
    """Stated confidence vs measured accuracy, per bucket.

    This is the metric that catches a model saying 0.95 and being right 70% of
    the time. Buckets with fewer than five decisive cases report `n` and no
    accuracy -- an accuracy computed on two cases is noise dressed as a finding.
    """
    bounds = [0.0, *edges, 1.01]
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        rows = [r for r in results if lo <= r.confidence < hi]
        scored = [r.agrees for r in rows if r.agrees is not None]
        out.append(
            {
                "range": f"[{lo:.2f}, {hi:.2f})",
                "n": len(rows),
                "n_decisive": len(scored),
                "mean_stated_confidence": round(
                    statistics.fmean([r.confidence for r in rows]), 4
                )
                if rows
                else None,
                "measured_accuracy": round(sum(scored) / len(scored), 4)
                if len(scored) >= 5
                else None,
            }
        )
    return out


def compare_to_floor(agent: dict[str, Any], floor: dict[str, Any]) -> dict[str, Any]:
    """The delta that justifies the LLM's cost.

    An agent scoring 0.86 where the rule engine scores 0.84 is not worth
    $0.0012 a case. An agent scoring 0.86 where the floor scores 0.62 -- almost
    all of the gap on the ambiguous stratum -- is.
    """

    def delta(path: list[str]) -> float | None:
        a, f = agent, floor
        for k in path:
            a = (a or {}).get(k)  # type: ignore[assignment]
            f = (f or {}).get(k)  # type: ignore[assignment]
        if a is None or f is None:
            return None
        return round(float(a) - float(f), 4)

    per_stratum = {}
    for s in agent.get("per_stratum", {}):
        per_stratum[s] = {
            "agent": agent["per_stratum"][s].get("agreement"),
            "floor": floor.get("per_stratum", {}).get(s, {}).get("agreement"),
            "delta": delta(["per_stratum", s, "agreement"]),
        }

    return {
        "agreement_macro_delta": delta(["agreement_macro"]),
        "decisive_rate_delta": delta(["decisive_rate"]),
        "narrative_score_delta": delta(["narrative", "mean_score"]),
        "cost_delta_per_1000_usd": delta(["cost", "per_1000_triages_usd"]),
        "per_stratum": per_stratum,
    }


def variance_report(runs: list[dict[str, Any]], keys: tuple[str, ...] = ("agreement_macro", "decisive_rate")) -> dict[str, Any]:
    """mean +- std across repeated runs of the whole suite.

    Publishing this is what stops a 2% improvement from being reported as
    progress when the run-to-run spread is 3%.
    """
    out: dict[str, Any] = {"runs": len(runs)}
    for k in keys:
        vals = [float(r[k]) for r in runs if r.get(k) is not None]
        if not vals:
            continue
        out[k] = {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4),
            "runs": [round(v, 4) for v in vals],
        }
    return out


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr[-1]), 2),
    }
