"""Cost dashboard and the daily self-report (capability C8).

The report answers the four questions a fraud engineering lead asks each
morning, and it answers them from measured data rather than from the constants
in the cost matrix:

1. What did we route yesterday, and did the funnel move?
2. Did quality hold -- agreement, hallucinations, repair rate?
3. What did it cost, and are we tracking to budget?
4. Did anything drift?

The drift check is the one that earns the job. A fraud scorer decays because the
world changes, not because the code did, and the first visible symptom is a
shift in the *score distribution* -- which shows up days before anyone notices
recall falling, because chargebacks take weeks to land. Comparing yesterday's
score histogram to the training reference via population stability index is a
cheap early warning, and PSI's conventional bands (0.1 / 0.25) give it a
threshold that does not need tuning.

`cost_per_triage` measured here is compared against `settings.triage_llm_usd`,
the constant the band optimiser assumed. When they diverge past 25%, the bands
were chosen against a price that no longer holds and the report says so. That
loop -- measured cost feeding back into threshold selection -- is the part most
systems never close.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from fraudsight.config import Settings, settings as default_settings

# Population Stability Index bands, the conventional risk-modelling thresholds.
PSI_SHIFT = 0.10  # worth looking at
PSI_ALERT = 0.25  # act


@dataclass
class DailyReport:
    day: str
    payload: dict[str, Any]

    def render(self) -> str:
        p = self.payload
        funnel = p["funnel"]
        cost = p["cost"]
        quality = p["quality"]
        drift = p["drift"]

        lines = [
            f"fraudsight daily report — {self.day}",
            "=" * 58,
            "",
            f"Volume            {funnel['transactions']:,} transactions",
        ]
        if funnel.get("complete"):
            lines += [
                f"  auto-approve    {funnel['rates'].get('auto_approve', 0):.1%}"
                f"  ({funnel['counts'].get('auto_approve', 0):,})",
                f"  triage          {funnel['rates'].get('triage', 0):.1%}"
                f"  ({funnel['counts'].get('triage', 0):,})",
                f"  auto-block      {funnel['rates'].get('auto_block', 0):.1%}"
                f"  ({funnel['counts'].get('auto_block', 0):,})",
                f"  briefs written  {funnel['briefs']:,}",
            ]
        else:
            lines += [
                f"  briefs written  {funnel['briefs']:,}",
                "  (band split unavailable — only triage runs are traced; pass",
                "   --band-counts or run `fraudsight replay` for the full funnel)",
            ]
        lines += [
            "",
            f"Quality",
            f"  schema pass     {quality['schema_pass_rate']:.1%}"
            f"  (first attempt {quality['first_attempt_rate']:.1%})",
            f"  repairs/brief   {quality['mean_repairs']:.2f}",
            f"  policy blocks   {quality['policy_block_rate']:.1%}",
            f"  escalation      {quality['escalation_rate']:.1%}",
            f"  degraded runs   {quality['degraded_rate']:.1%}",
            f"  injection hits  {quality['injection_attempts']}",
            "",
            f"Cost",
            f"  spend           ${cost['spend_usd']:.4f} of ${cost['budget_usd']:.2f}"
            f"  ({cost['budget_used_pct']:.0f}%)",
            f"  per triage      ${cost['per_triage_usd']:.6f}"
            f"  (assumed ${cost['assumed_per_triage_usd']:.6f})",
            f"  per 1k txns     ${cost['per_1000_transactions_usd']:.4f}",
            f"  cache hit rate  {cost['cache_hit_rate']:.1%}",
            "",
            f"Latency (triage)  p50 {p['latency'].get('p50_ms', 0):.0f}ms"
            f"  p95 {p['latency'].get('p95_ms', 0):.0f}ms"
            f"  p99 {p['latency'].get('p99_ms', 0):.0f}ms",
            "",
            f"Drift             score PSI {drift['score_psi']:.4f} — {drift['verdict']}",
        ]
        for alert in p["alerts"]:
            lines.append(f"  ALERT  {alert}")
        if not p["alerts"]:
            lines.append("  no alerts")
        return "\n".join(lines)


def build(
    traces: list[dict[str, Any]],
    *,
    reference_scores: np.ndarray | None = None,
    day: str | None = None,
    cfg: Settings | None = None,
    total_transactions: int | None = None,
    band_counts: dict[str, int] | None = None,
) -> DailyReport:
    cfg = cfg or default_settings
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")

    n_triage = len(traces)
    n_total = total_transactions or n_triage

    # Only triage runs produce traces -- auto-approved and auto-blocked
    # transactions never reach the agent, which is the entire point of the
    # design. So the funnel cannot be recovered from traces alone: counting
    # them reported "100% triage" on a run that auto-approved 86% of its
    # volume. `band_counts` comes from the scoring path (the replay harness
    # writes it), and without it the report says so rather than inventing a
    # split it cannot see.
    bands = Counter(band_counts) if band_counts else Counter(
        t.get("band", "triage") for t in traces
    )
    funnel_complete = band_counts is not None

    scores = np.asarray([float(t.get("score", 0.0)) for t in traces], dtype=np.float64)

    # Drift has to compare like with like. Traces exist only for the triage
    # band, so scoring them against the full-population reference distribution
    # measures the band filter rather than any drift -- it produced a PSI of
    # 3.26 ("significant shift, investigate") on a system that had not shifted
    # at all. Restricting the reference to the same band is what makes the
    # number mean something.
    if reference_scores is not None and scores.size:
        lo, hi = float(scores.min()), float(scores.max())
        in_band = reference_scores[(reference_scores >= lo) & (reference_scores <= hi)]
        reference_scores = in_band if in_band.size >= 100 else reference_scores
    latencies = [float(t.get("total_ms", 0.0)) for t in traces]
    costs = [float(t.get("cost_usd", 0.0)) for t in traces]
    spend = sum(costs)

    schema_ok = sum(1 for t in traces if t.get("recommendation"))
    first_ok = sum(1 for t in traces if t.get("recommendation") and not t.get("repairs"))
    policy_blocked = sum(1 for t in traces if t.get("policy_violations"))
    escalated = sum(
        1 for t in traces if t.get("recommendation") == "escalate_to_analyst"
    )
    degraded = sum(1 for t in traces if t.get("degraded"))
    injections = sum(1 for t in traces if t.get("injection_detected"))

    cache_read = sum(int(t.get("cache_read_tokens", 0)) for t in traces)
    input_tokens = sum(int(t.get("input_tokens", 0)) for t in traces)

    per_triage = spend / max(n_triage, 1)
    psi = (
        population_stability_index(reference_scores, scores)
        if reference_scores is not None and scores.size
        else 0.0
    )

    alerts: list[str] = []
    if spend >= cfg.daily_llm_budget_usd * cfg.budget_alert_fraction:
        alerts.append(
            f"spend ${spend:.4f} is past {cfg.budget_alert_fraction:.0%} of the "
            f"${cfg.daily_llm_budget_usd:.2f} daily budget"
        )
    if psi >= PSI_ALERT:
        alerts.append(
            f"score distribution PSI {psi:.3f} exceeds {PSI_ALERT} — retrain or "
            f"re-tune the bands"
        )
    if n_triage and degraded / n_triage > 0.05:
        alerts.append(
            f"{degraded / n_triage:.1%} of triages ran degraded (fallback provider)"
        )
    if n_triage and policy_blocked / n_triage > 0.10:
        alerts.append(f"{policy_blocked / n_triage:.1%} of briefs tripped output policy")
    assumed = cfg.triage_llm_usd
    if n_triage and assumed > 0 and abs(per_triage - assumed) / assumed > 0.25:
        alerts.append(
            f"measured cost per triage ${per_triage:.6f} differs from the ${assumed:.6f} "
            f"assumed by the band optimiser by more than 25% — re-run `fraudsight thresholds`"
        )
    if injections:
        alerts.append(f"{injections} prompt-injection attempt(s) neutralised")

    payload = {
        "funnel": {
            "transactions": n_total,
            "counts": dict(bands),
            "rates": {k: round(v / max(n_total, 1), 6) for k, v in bands.items()},
            "briefs": n_triage,
            "complete": funnel_complete,
        },
        "quality": {
            "schema_pass_rate": round(schema_ok / max(n_triage, 1), 4),
            "first_attempt_rate": round(first_ok / max(n_triage, 1), 4),
            "mean_repairs": round(
                statistics.fmean([int(t.get("repairs", 0)) for t in traces]), 3
            )
            if traces
            else 0.0,
            "policy_block_rate": round(policy_blocked / max(n_triage, 1), 4),
            "escalation_rate": round(escalated / max(n_triage, 1), 4),
            "degraded_rate": round(degraded / max(n_triage, 1), 4),
            "injection_attempts": injections,
            "recommendations": dict(
                Counter(t.get("recommendation") for t in traces if t.get("recommendation"))
            ),
        },
        "cost": {
            "spend_usd": round(spend, 6),
            "budget_usd": cfg.daily_llm_budget_usd,
            "budget_used_pct": round(100 * spend / max(cfg.daily_llm_budget_usd, 1e-9), 2),
            "per_triage_usd": round(per_triage, 8),
            "assumed_per_triage_usd": cfg.triage_llm_usd,
            "per_1000_transactions_usd": round(1000 * spend / max(n_total, 1), 6),
            "per_1000_alerts_usd": round(1000 * per_triage, 6),
            "cache_hit_rate": round(cache_read / max(cache_read + input_tokens, 1), 4),
            "tokens": {"input": input_tokens, "cache_read": cache_read},
        },
        "latency": _percentiles(latencies),
        "drift": {
            "score_psi": round(psi, 6),
            "verdict": _psi_verdict(psi),
            "reference_n": int(reference_scores.size) if reference_scores is not None else 0,
            "observed_n": int(scores.size),
        },
        "alerts": alerts,
    }
    return DailyReport(day=day, payload=payload)


def population_stability_index(
    reference: np.ndarray, observed: np.ndarray, bins: int = 10
) -> float:
    """PSI over score deciles of the reference distribution.

    Quantile bins from the *reference*, not from the observed data: binning on
    the observed sample would move the goalposts with the very shift the metric
    is meant to detect.
    """
    if reference.size == 0 or observed.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    obs_counts, _ = np.histogram(observed, bins=edges)
    # Laplace smoothing: an empty bucket makes the log term infinite, and a
    # single empty decile should not report infinite drift.
    ref_p = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    obs_p = (obs_counts + 1) / (obs_counts.sum() + len(obs_counts))
    return float(np.sum((obs_p - ref_p) * np.log(obs_p / ref_p)))


def _psi_verdict(psi: float) -> str:
    if psi < PSI_SHIFT:
        return "stable"
    if psi < PSI_ALERT:
        return f"moderate shift (>{PSI_SHIFT}) — watch"
    return f"significant shift (>{PSI_ALERT}) — investigate"


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


def save(report: DailyReport, cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    d = cfg.artifact_dir / "reports"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"daily_{report.day}.json"
    path.write_text(
        json.dumps({"day": report.day, **report.payload}, indent=2), encoding="utf-8"
    )
    (d / "daily_latest.txt").write_text(report.render(), encoding="utf-8")
    return path


def yesterday() -> str:
    return (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
