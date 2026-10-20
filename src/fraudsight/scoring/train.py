"""Train the hot-path scorer: LightGBM + isotonic calibration.

Methodology notes, because the numbers are only worth anything if the protocol
holds up:

* **AUC-PR, not AUC-ROC.** At a 3.5% positive rate, ROC-AUC is dominated by the
  enormous negative class and reads ~0.97 for a model that is mediocre where it
  matters. Both are reported; AUC-PR is the one that moves with real quality.

* **val is split chronologically in half.** The early half drives early
  stopping, the late half fits the calibrator and selects the bands. Reusing one
  set for all three gives probabilities that look sharper than they are.

* **test is touched once**, to produce the final table. No threshold, no
  hyperparameter, and no feature decision is made against it.

* **3 seeds, mean +- std.** A 2% improvement means nothing next to a 3%
  run-to-run spread, so the spread is published alongside the point estimate.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraudsight.config import settings
from fraudsight.features.spec import CATEGORICAL_COLS
from fraudsight.scoring.calibration import (
    IsotonicCalibrator,
    expected_calibration_error,
    reliability_curve,
)
from fraudsight.scoring.dataset import Split, artifact, load_split, load_stats

SEEDS = (13, 29, 41)


@dataclass
class TrainConfig:
    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 64
    min_child_samples: int = 60
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.65
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 100
    # Rebalancing to the class ratio (~28:1) rather than a hand-tuned value.
    # LightGBM's `is_unbalance` does the same thing less explicitly.
    use_scale_pos_weight: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _params(cfg: TrainConfig, seed: int, pos_weight: float) -> dict[str, Any]:
    p: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "subsample_freq": cfg.subsample_freq,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "n_jobs": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if cfg.use_scale_pos_weight:
        p["scale_pos_weight"] = pos_weight
    p.update(cfg.extra)
    return p


def _score_metrics(probs: NDArray[np.float64], y: NDArray[np.int_]) -> dict[str, float]:
    return {
        "auc_pr": round(float(average_precision_score(y, probs)), 6),
        "auc_roc": round(float(roc_auc_score(y, probs)), 6),
        "brier": round(float(brier_score_loss(y, probs)), 6),
        "ece": expected_calibration_error(probs, y),
        "recall_at_1pct_fpr": _recall_at_fpr(probs, y, 0.01),
        "recall_at_5pct_fpr": _recall_at_fpr(probs, y, 0.05),
        "precision_at_50pct_recall": _precision_at_recall(probs, y, 0.50),
        "base_rate": round(float(y.mean()), 6),
    }


def _recall_at_fpr(probs: NDArray[np.float64], y: NDArray[np.int_], target_fpr: float) -> float:
    """Recall achievable while holding the false-positive rate at target.

    This is the operationally meaningful number: FPR is what determines how many
    genuine customers get inconvenienced, and it is the constraint the business
    actually sets.
    """
    neg = probs[y == 0]
    if neg.size == 0:
        return 0.0
    cut = float(np.quantile(neg, 1.0 - target_fpr))
    pos = probs[y == 1]
    return round(float((pos >= cut).mean()), 6) if pos.size else 0.0


def _precision_at_recall(
    probs: NDArray[np.float64], y: NDArray[np.int_], target_recall: float
) -> float:
    pos = probs[y == 1]
    if pos.size == 0:
        return 0.0
    cut = float(np.quantile(pos, 1.0 - target_recall))
    flagged = probs >= cut
    n = int(flagged.sum())
    return round(float(y[flagged].mean()), 6) if n else 0.0


def _latency_profile(
    booster: lgb.Booster, cal: IsotonicCalibrator, X: Any, n: int = 400
) -> dict[str, float]:
    """Single-row inference latency, which is what the SLA is written against.

    Batch throughput is easy and irrelevant: production scores one transaction
    at a time on the request path.
    """
    rows = [X.iloc[[i]] for i in range(min(n, len(X)))]
    for r in rows[:20]:  # warm up the predictor
        booster.predict(r, raw_score=True)
    samples: list[float] = []
    for r in rows:
        t0 = time.perf_counter()
        raw = booster.predict(r, raw_score=True)
        cal.transform(np.asarray(raw, dtype=np.float64))
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()

    def q(p: float) -> float:
        return round(samples[min(int(p * len(samples)), len(samples) - 1)], 3)

    return {
        "n_samples": len(samples),
        "p50_ms": q(0.50),
        "p95_ms": q(0.95),
        "p99_ms": q(0.99),
        "mean_ms": round(sum(samples) / len(samples), 3),
        "max_ms": round(samples[-1], 3),
    }


@dataclass
class RunResult:
    seed: int
    best_iteration: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    booster: lgb.Booster
    calibrator: IsotonicCalibrator
    train_seconds: float


def train_one(
    train: Split,
    val_early: Split,
    val_late: Split,
    test: Split,
    seed: int,
    cfg: TrainConfig,
) -> RunResult:
    pos = max(int(train.y.sum()), 1)
    neg = len(train) - pos
    params = _params(cfg, seed, pos_weight=neg / pos)

    cats = list(CATEGORICAL_COLS)
    dtrain = lgb.Dataset(train.X, label=train.y, categorical_feature=cats, free_raw_data=False)
    dvalid = lgb.Dataset(
        val_early.X,
        label=val_early.y,
        categorical_feature=cats,
        reference=dtrain,
        free_raw_data=False,
    )

    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        valid_sets=[dvalid],
        valid_names=["val_early"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_seconds = time.perf_counter() - t0

    # Calibrate on the late half of val -- disjoint from the rows that chose the
    # iteration count above.
    raw_late = booster.predict(val_late.X, raw_score=True)
    cal = IsotonicCalibrator.fit(np.asarray(raw_late, dtype=np.float64), val_late.y)

    val_probs = cal.transform(np.asarray(raw_late, dtype=np.float64))
    raw_test = np.asarray(booster.predict(test.X, raw_score=True), dtype=np.float64)
    test_probs = cal.transform(raw_test)

    return RunResult(
        seed=seed,
        best_iteration=int(booster.best_iteration or cfg.n_estimators),
        val_metrics=_score_metrics(val_probs, val_late.y),
        test_metrics=_score_metrics(test_probs, test.y),
        booster=booster,
        calibrator=cal,
        train_seconds=round(train_seconds, 1),
    )


def _aggregate(runs: list[RunResult], which: str) -> dict[str, dict[str, float]]:
    keys = runs[0].val_metrics if which == "val" else runs[0].test_metrics
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [getattr(r, f"{which}_metrics")[k] for r in runs]
        out[k] = {
            "mean": round(statistics.fmean(vals), 6),
            "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 6),
            "runs": [round(v, 6) for v in vals],
        }
    return out


def train(seeds: tuple[int, ...] = SEEDS, cfg: TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    settings.ensure_dirs()
    stats = load_stats()

    train_split = load_split("train", stats)
    val_split = load_split("val", stats)
    test_split = load_split("test", stats)
    val_early, val_late = val_split.split_by_time_median()

    runs = [train_one(train_split, val_early, val_late, test_split, s, cfg) for s in seeds]

    # Ship the median-AUC-PR model, not the best. Picking the best seed on test
    # is a subtle form of test-set selection; the median is what a fresh seed
    # would plausibly reproduce.
    ranked = sorted(runs, key=lambda r: r.val_metrics["auc_pr"])
    chosen = ranked[len(ranked) // 2]

    chosen.booster.save_model(str(artifact("model.txt")), num_iteration=chosen.best_iteration)
    chosen.calibrator.save(artifact("calibrator.json"))

    raw_test = np.asarray(chosen.booster.predict(test_split.X, raw_score=True), dtype=np.float64)
    test_probs = chosen.calibrator.transform(raw_test)
    np.save(artifact("test_probs.npy"), test_probs)
    np.save(artifact("test_labels.npy"), test_split.y)

    raw_val = np.asarray(chosen.booster.predict(val_late.X, raw_score=True), dtype=np.float64)
    np.save(artifact("val_probs.npy"), chosen.calibrator.transform(raw_val))
    np.save(artifact("val_labels.npy"), val_late.y)

    gain = chosen.booster.feature_importance(importance_type="gain")
    names = chosen.booster.feature_name()
    top = sorted(zip(names, gain, strict=True), key=lambda kv: -kv[1])[:30]

    report: dict[str, Any] = {
        "config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "seeds": list(seeds),
        "chosen_seed": chosen.seed,
        "selection_rule": "median val AUC-PR across seeds",
        "data": {
            "train_rows": len(train_split),
            "train_fraud": train_split.n_positive,
            "val_early_rows": len(val_early),
            "val_late_rows": len(val_late),
            "val_late_fraud": val_late.n_positive,
            "test_rows": len(test_split),
            "test_fraud": test_split.n_positive,
            "features": int(train_split.X.shape[1]),
            "categorical_features": len(CATEGORICAL_COLS),
        },
        "best_iterations": [r.best_iteration for r in runs],
        "train_seconds": [r.train_seconds for r in runs],
        "val_metrics": _aggregate(runs, "val"),
        "test_metrics": _aggregate(runs, "test"),
        "calibration": {
            "method": "isotonic",
            "fit_rows": chosen.calibrator.n_fit_rows,
            "fit_positives": chosen.calibrator.n_fit_positives,
            "knots": len(chosen.calibrator.x_knots),
            "test_reliability": reliability_curve(test_probs, test_split.y, n_bins=10),
        },
        "latency": _latency_profile(chosen.booster, chosen.calibrator, test_split.X),
        "top_features_by_gain": [{"feature": n, "gain": round(float(g), 1)} for n, g in top],
    }

    (artifact("scorer_metrics.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
