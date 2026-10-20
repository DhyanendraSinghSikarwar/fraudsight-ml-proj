"""Score-band selection as expected-cost minimisation.

This is the module that turns an ML metric into a business decision. Picking
bands by maximising F1 optimises a quantity nobody in the business cares about
and implicitly prices a missed fraud the same as a wrongly declined customer.
They are not the same: at the defaults here a missed fraud costs $180 and a
false decline costs $25, so the optimiser will happily trade precision for
recall -- which is the correct behaviour and the reason recall is the one
number we refuse to trade away.

Three outcomes per transaction, one cost each:

  auto-approve (score < lo)     fraud -> fraud_loss        legit -> $0
  triage       (lo <= s < hi)   analyst review + LLM cost, decision assumed correct
  auto-block   (score >= hi)    fraud -> $0               legit -> false_decline

The triage band is not free: it costs an analyst's time on every case. That is
what stops the optimiser from routing everything to humans.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CostMatrix:
    fraud_loss_usd: float
    false_decline_usd: float
    analyst_review_usd: float
    triage_llm_usd: float

    # Analysts are not oracles. Measured agreement between an analyst working
    # from a brief and the eventual chargeback outcome; 0.92 is conservative
    # against published industry numbers. Setting this to 1.0 would make the
    # triage band look free and skew the bands toward routing everything.
    analyst_accuracy: float = 0.92


@dataclass(frozen=True)
class BandOutcome:
    lo: float
    hi: float
    expected_cost_usd: float
    cost_per_txn_usd: float
    recall: float
    precision_blocked: float
    review_rate: float
    auto_approve_rate: float
    auto_block_rate: float
    fraud_missed: int
    fraud_caught_block: int
    fraud_in_triage: int
    legit_blocked: int
    n: int


def evaluate_bands(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    lo: float,
    hi: float,
    costs: CostMatrix,
) -> BandOutcome:
    """Total expected cost of routing `probs` with bands (lo, hi)."""
    n = int(labels.size)
    approve = probs < lo
    block = probs >= hi
    triage = ~approve & ~block

    fraud = labels == 1
    legit = ~fraud

    fraud_missed = int((approve & fraud).sum())
    fraud_caught_block = int((block & fraud).sum())
    fraud_in_triage = int((triage & fraud).sum())
    legit_blocked = int((block & legit).sum())
    n_triage = int(triage.sum())

    # An analyst working a triage case is right `analyst_accuracy` of the time.
    # A wrong call on a fraud case costs the fraud loss; a wrong call on a legit
    # case costs a false decline.
    miss_rate = 1.0 - costs.analyst_accuracy
    triage_fraud_leak = fraud_in_triage * miss_rate * costs.fraud_loss_usd
    triage_legit_harm = (n_triage - fraud_in_triage) * miss_rate * costs.false_decline_usd

    total = (
        fraud_missed * costs.fraud_loss_usd
        + legit_blocked * costs.false_decline_usd
        + n_triage * (costs.analyst_review_usd + costs.triage_llm_usd)
        + triage_fraud_leak
        + triage_legit_harm
    )

    total_fraud = int(fraud.sum())
    # Recall = fraud that does NOT get auto-approved, i.e. everything the system
    # either blocks or escalates. Auto-approval is the only unrecoverable path.
    caught = fraud_caught_block + fraud_in_triage
    recall = caught / total_fraud if total_fraud else 0.0
    n_block = int(block.sum())

    return BandOutcome(
        lo=round(lo, 6),
        hi=round(hi, 6),
        expected_cost_usd=round(total, 2),
        cost_per_txn_usd=round(total / n, 6) if n else 0.0,
        recall=round(recall, 6),
        precision_blocked=round(fraud_caught_block / n_block, 6) if n_block else 0.0,
        review_rate=round(n_triage / n, 6) if n else 0.0,
        auto_approve_rate=round(int(approve.sum()) / n, 6) if n else 0.0,
        auto_block_rate=round(n_block / n, 6) if n else 0.0,
        fraud_missed=fraud_missed,
        fraud_caught_block=fraud_caught_block,
        fraud_in_triage=fraud_in_triage,
        legit_blocked=legit_blocked,
        n=n,
    )


def optimise(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    costs: CostMatrix,
    min_recall: float = 0.95,
    max_review_rate: float = 0.15,
    grid: int = 60,
) -> dict[str, object]:
    """Grid-search (lo, hi) minimising expected cost subject to hard constraints.

    `min_recall` is a constraint, not a term in the objective. Expressing it as
    a penalty would let a large enough cost saving buy its way through the
    recall floor, and in a fraud system that floor is a commitment to the
    business, not a preference.
    """
    # Search lo on the quantile grid of the score distribution: a uniform grid
    # wastes almost every point in a region where 3.5% of the mass lives.
    lo_grid = np.unique(np.quantile(probs, np.linspace(0.50, 0.995, grid)))
    hi_grid = np.unique(np.quantile(probs, np.linspace(0.90, 0.99999, grid)))

    best: BandOutcome | None = None
    feasible = 0
    considered = 0
    all_rows: list[BandOutcome] = []

    for lo in lo_grid:
        for hi in hi_grid:
            if hi <= lo:
                continue
            considered += 1
            out = evaluate_bands(probs, labels, float(lo), float(hi), costs)
            all_rows.append(out)
            if out.recall < min_recall or out.review_rate > max_review_rate:
                continue
            feasible += 1
            if best is None or out.expected_cost_usd < best.expected_cost_usd:
                best = out

    if best is None:
        raise ValueError(
            f"no (lo, hi) satisfies recall >= {min_recall} and review_rate <= "
            f"{max_review_rate}. Loosen max_review_rate or improve the scorer -- "
            f"do not lower the recall floor."
        )

    # Baselines the chosen bands have to beat, for the README results table.
    score_only = _best_single_threshold(probs, labels, costs, min_recall)
    review_all = evaluate_bands(probs, labels, 0.0, 1.01, costs)
    approve_all = evaluate_bands(probs, labels, 1.01, 1.02, costs)

    return {
        "chosen": asdict(best),
        "constraints": {"min_recall": min_recall, "max_review_rate": max_review_rate},
        "costs": asdict(costs),
        "search": {"considered": considered, "feasible": feasible, "grid": grid},
        "baselines": {
            "block_only_no_triage": asdict(score_only),
            "review_everything": asdict(review_all),
            "approve_everything": asdict(approve_all),
        },
        "savings_vs_review_everything_usd": round(
            review_all.expected_cost_usd - best.expected_cost_usd, 2
        ),
        "savings_vs_block_only_usd": round(score_only.expected_cost_usd - best.expected_cost_usd, 2),
        "cost_surface": [asdict(r) for r in _thin(all_rows, 400)],
    }


def frontier(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    costs: CostMatrix,
    recall_targets: tuple[float, ...] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99),
    grid: int = 60,
) -> list[dict[str, object]]:
    """Cheapest bands that reach each recall target, with no review-rate cap.

    This exists because the recall floor in the NFR table turned out not to be
    reachable at a defensible review rate on this scorer, and the useful
    response to that is a published curve rather than a quietly relaxed
    constant. Each row answers "what would recall R actually cost?", so the
    operating point becomes an argument someone can disagree with using numbers.

    Rows where the target is unreachable at any (lo, hi) are returned with
    `feasible: False` rather than omitted -- a gap in the curve is information.
    """
    lo_grid = np.unique(np.quantile(probs, np.linspace(0.0, 0.995, grid)))
    hi_grid = np.unique(np.quantile(probs, np.linspace(0.90, 0.99999, grid)))

    rows: list[dict[str, object]] = []
    for target in recall_targets:
        best: BandOutcome | None = None
        for lo in lo_grid:
            for hi in hi_grid:
                if hi <= lo:
                    continue
                out = evaluate_bands(probs, labels, float(lo), float(hi), costs)
                if out.recall < target:
                    continue
                if best is None or out.expected_cost_usd < best.expected_cost_usd:
                    best = out
        if best is None:
            rows.append({"recall_target": target, "feasible": False})
            continue
        rows.append(
            {
                "recall_target": target,
                "feasible": True,
                "achieved_recall": best.recall,
                "review_rate": best.review_rate,
                "auto_block_rate": best.auto_block_rate,
                "cost_per_txn_usd": best.cost_per_txn_usd,
                "expected_cost_usd": best.expected_cost_usd,
                "lo": best.lo,
                "hi": best.hi,
                "fraud_missed": best.fraud_missed,
                "legit_blocked": best.legit_blocked,
            }
        )
    return rows


def _best_single_threshold(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    costs: CostMatrix,
    min_recall: float,
) -> BandOutcome:
    """No-triage baseline: one cut, approve below / block above. This is the
    system you ship if you skip the LLM layer entirely, and it is what the
    triage band has to justify itself against."""
    best: BandOutcome | None = None
    for t in np.unique(np.quantile(probs, np.linspace(0.50, 0.9999, 200))):
        out = evaluate_bands(probs, labels, float(t), float(t), costs)
        if out.recall < min_recall:
            continue
        if best is None or out.expected_cost_usd < best.expected_cost_usd:
            best = out
    if best is None:  # recall floor unreachable without a triage band
        return evaluate_bands(probs, labels, 0.0, 0.0, costs)
    return best


def select_operating_point(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    costs: CostMatrix,
    *,
    min_recall: float,
    max_review_rate: float,
    grid: int = 60,
) -> dict[str, object]:
    """Choose bands, and be explicit when the recall NFR cannot be met.

    The order of operations is the point. We ask for the constrained optimum
    first; only if no (lo, hi) satisfies the floor do we fall back to the
    unconstrained cost minimum -- and then the report carries
    `nfr_recall_met: False` and the frontier that shows what the floor would
    have cost. Nothing silently relaxes the constraint, and nothing refuses to
    ship because a target was missed.
    """
    curve = frontier(probs, labels, costs, grid=grid)
    try:
        report = optimise(
            probs, labels, costs, min_recall=min_recall, max_review_rate=max_review_rate, grid=grid
        )
        report["nfr_recall_met"] = True
        report["selection"] = "expected-cost minimum subject to the recall floor"
    except ValueError as exc:
        report = optimise(probs, labels, costs, min_recall=0.0, max_review_rate=1.0, grid=grid)
        report["nfr_recall_met"] = False
        report["nfr_shortfall"] = str(exc)
        report["selection"] = (
            "expected-cost minimum with the recall floor RELAXED -- the floor was "
            "unreachable at any review rate under the cap; see docs/scope.md"
        )
    report["frontier"] = curve
    report["requested_constraints"] = {
        "min_recall": min_recall,
        "max_review_rate": max_review_rate,
    }
    return report


def _thin(rows: list[BandOutcome], limit: int) -> list[BandOutcome]:
    if len(rows) <= limit:
        return rows
    step = len(rows) // limit
    return rows[::step][:limit]


def save(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_bands(path: Path) -> tuple[float, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    chosen = report["chosen"]
    return float(chosen["lo"]), float(chosen["hi"])
