"""Isotonic calibration, stored as interpolation knots.

Why not `CalibratedClassifierCV`: the serving path needs to apply calibration
in a few microseconds with no sklearn import, and an auditor needs to be able
to read the mapping. Fitting isotonic regression and persisting the knots gives
both -- serving is a single `np.interp`, and artifacts/calibrator.json is a
plain monotone table anyone can inspect.

Calibration is not cosmetic here. The routing bands are dollar thresholds
derived from a cost matrix, and a cost matrix applied to uncalibrated scores is
meaningless: "0.95" has to actually mean "95% of these are fraud".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone piecewise-linear map from raw model score to calibrated P(fraud)."""

    x_knots: tuple[float, ...]
    y_knots: tuple[float, ...]
    n_fit_rows: int
    n_fit_positives: int

    @classmethod
    def fit(cls, raw_scores: NDArray[np.float64], labels: NDArray[np.int_]) -> IsotonicCalibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_scores, labels)
        return cls(
            x_knots=tuple(float(v) for v in iso.X_thresholds_),
            y_knots=tuple(float(v) for v in iso.y_thresholds_),
            n_fit_rows=int(labels.size),
            n_fit_positives=int(labels.sum()),
        )

    def transform(self, raw_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(
            raw_scores,
            np.asarray(self.x_knots, dtype=np.float64),
            np.asarray(self.y_knots, dtype=np.float64),
            left=self.y_knots[0],
            right=self.y_knots[-1],
        )

    def transform_scalar(self, raw_score: float) -> float:
        return float(self.transform(np.asarray([raw_score], dtype=np.float64))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "x_knots": list(self.x_knots),
                    "y_knots": list(self.y_knots),
                    "n_fit_rows": self.n_fit_rows,
                    "n_fit_positives": self.n_fit_positives,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            x_knots=tuple(d["x_knots"]),
            y_knots=tuple(d["y_knots"]),
            n_fit_rows=int(d["n_fit_rows"]),
            n_fit_positives=int(d["n_fit_positives"]),
        )


def reliability_curve(
    probs: NDArray[np.float64], labels: NDArray[np.int_], n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-count (quantile) bins. Equal-width bins put ~97% of a 3.5%-positive
    portfolio in the first bucket and tell you nothing about the tail, which is
    the only part the routing bands care about.
    """
    order = np.argsort(probs)
    p_sorted, y_sorted = probs[order], labels[order]
    bins = np.array_split(np.arange(p_sorted.size), n_bins)
    out: list[dict[str, float]] = []
    for idx in bins:
        if idx.size == 0:
            continue
        out.append(
            {
                "n": int(idx.size),
                "predicted_mean": round(float(p_sorted[idx].mean()), 6),
                "observed_rate": round(float(y_sorted[idx].mean()), 6),
                "p_low": round(float(p_sorted[idx].min()), 6),
                "p_high": round(float(p_sorted[idx].max()), 6),
            }
        )
    return out


def expected_calibration_error(
    probs: NDArray[np.float64], labels: NDArray[np.int_], n_bins: int = 10
) -> float:
    curve = reliability_curve(probs, labels, n_bins=n_bins)
    total = sum(b["n"] for b in curve)
    if total == 0:
        return 0.0
    return round(
        sum(b["n"] / total * abs(b["predicted_mean"] - b["observed_rate"]) for b in curve), 6
    )
