"""Band selection must behave like a cost model, not like an accuracy metric."""

from __future__ import annotations

import numpy as np
import pytest

from fraudsight.scoring.thresholds import (
    CostMatrix,
    evaluate_bands,
    frontier,
    optimise,
    select_operating_point,
)

COSTS = CostMatrix(
    fraud_loss_usd=180.0, false_decline_usd=25.0, analyst_review_usd=4.20, triage_llm_usd=0.0012
)


def _synthetic(n: int = 4_000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A separable-but-overlapping score distribution, like a real scorer."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.035).astype(int)
    probs = np.where(
        labels == 1,
        rng.beta(5, 4, n),  # fraud skews high
        rng.beta(1.2, 20, n),  # legitimate crowds near zero
    )
    return np.clip(probs, 0.0, 1.0), labels


def test_recall_counts_triage_as_caught() -> None:
    """Only auto-approval loses a fraud. Escalation is a catch, not a miss --
    getting this backwards would make the triage band look worthless."""
    probs = np.array([0.01, 0.40, 0.99])
    labels = np.array([1, 1, 1])
    out = evaluate_bands(probs, labels, 0.05, 0.95, COSTS)
    assert out.fraud_missed == 1
    assert out.fraud_in_triage == 1
    assert out.fraud_caught_block == 1
    assert out.recall == pytest.approx(2 / 3)


def test_triage_band_is_not_free() -> None:
    """Routing everything to analysts must cost more than routing nothing,
    otherwise the optimiser would send the whole book to review."""
    probs, labels = _synthetic()
    review_all = evaluate_bands(probs, labels, 0.0, 1.01, COSTS)
    assert review_all.review_rate == pytest.approx(1.0)
    assert review_all.expected_cost_usd > len(labels) * COSTS.analyst_review_usd * 0.9


def test_optimiser_beats_both_trivial_baselines() -> None:
    probs, labels = _synthetic()
    report = optimise(probs, labels, COSTS, min_recall=0.0, max_review_rate=1.0, grid=25)
    chosen = report["chosen"]["expected_cost_usd"]
    assert chosen < report["baselines"]["review_everything"]["expected_cost_usd"]
    assert chosen < report["baselines"]["approve_everything"]["expected_cost_usd"]
    assert chosen <= report["baselines"]["block_only_no_triage"]["expected_cost_usd"]


def test_infeasible_recall_raises_rather_than_silently_relaxing() -> None:
    """The floor is a commitment. Quietly lowering it is the failure this
    guards against."""
    probs, labels = _synthetic()
    with pytest.raises(ValueError, match="do not lower the recall floor"):
        optimise(probs, labels, COSTS, min_recall=0.999, max_review_rate=0.01, grid=20)


def test_select_operating_point_reports_the_miss_and_still_ships() -> None:
    probs, labels = _synthetic()
    report = select_operating_point(
        probs, labels, COSTS, min_recall=0.999, max_review_rate=0.01, grid=20
    )
    assert report["nfr_recall_met"] is False
    assert "nfr_shortfall" in report
    assert report["chosen"]["lo"] < report["chosen"]["hi"]
    assert report["frontier"]


def test_frontier_is_monotone_in_cost() -> None:
    """Higher recall cannot be cheaper. If it is, the search is broken."""
    probs, labels = _synthetic()
    rows = [r for r in frontier(probs, labels, COSTS, (0.6, 0.8, 0.95), grid=25) if r["feasible"]]
    costs = [r["cost_per_txn_usd"] for r in rows]
    reviews = [r["review_rate"] for r in rows]
    assert costs == sorted(costs), f"cost fell as recall rose: {costs}"
    assert reviews == sorted(reviews), f"review rate fell as recall rose: {reviews}"


def test_cost_matrix_drives_the_choice() -> None:
    """Ten-fold the cost of a missed fraud and the optimiser must widen the net.
    If the bands do not move, the cost matrix is decorative."""
    probs, labels = _synthetic()
    cheap = optimise(probs, labels, COSTS, min_recall=0.0, max_review_rate=1.0, grid=25)
    expensive_fraud = optimise(
        probs,
        labels,
        CostMatrix(1_800.0, 25.0, 4.20, 0.0012),
        min_recall=0.0,
        max_review_rate=1.0,
        grid=25,
    )
    assert expensive_fraud["chosen"]["recall"] >= cheap["chosen"]["recall"]
    assert expensive_fraud["chosen"]["lo"] <= cheap["chosen"]["lo"]
