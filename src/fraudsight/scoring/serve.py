"""Hot-path scoring: one transaction, no pandas, no sklearn.

The training-time latency probe measures ~26ms p50 per row, and almost all of it
is `DataFrame.iloc[[i]]` -- constructing a one-row pandas frame allocates ~190
Series and re-blocks the result. LightGBM itself is not the cost. So the serving
path builds a `(1, n_features)` float32 array directly and hands that to the
booster, which is what takes the p99 under the 50ms NFR.

That optimisation introduces exactly one hazard, and it is the reason
`_encode_categorical` exists. When LightGBM is trained on a pandas frame with
`category` dtype it stores the category ordering and maps strings to codes for
you. Predict on a raw numpy array and it assumes the caller already did that
mapping -- with no error if the caller got it wrong, just quietly different
codes and a plausible-looking score. The vocabulary is pinned in
`EntityStats.category_vocab`, this module encodes against it, and
tests/unit/test_serving_parity.py asserts the fast path and the pandas path
agree to 1e-9 on real rows. That test is the only thing standing between this
speed-up and a silent scoring bug.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fraudsight.config import settings
from fraudsight.features.spec import (
    CATEGORICAL_COLS,
    DERIVED_NUMERIC,
    PASSTHROUGH_NUMERIC,
    EntityStats,
    feature_names,
)
from fraudsight.scoring.calibration import IsotonicCalibrator


class Band(StrEnum):
    AUTO_APPROVE = "auto_approve"
    TRIAGE = "triage"
    AUTO_BLOCK = "auto_block"


@dataclass(frozen=True)
class ScoreResult:
    transaction_id: str
    score: float
    raw_score: float
    band: Band
    lo: float
    hi: float
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "score": round(self.score, 6),
            "raw_score": round(self.raw_score, 6),
            "band": str(self.band),
            "bands": {"auto_approve_below": self.lo, "auto_block_above": self.hi},
            "latency_ms": self.latency_ms,
        }


@dataclass
class LatencyRecorder:
    """Reservoir-free percentile tracking: keep every sample, sort on read.

    Fine at portfolio scale (a full replay is ~84k samples, ~700KB) and exact,
    which matters more here than memory -- an approximate p99 is not evidence
    for an SLA claim.
    """

    samples: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def percentiles(self) -> dict[str, float]:
        if not self.samples:
            return {}
        arr = np.sort(np.asarray(self.samples, dtype=np.float64))
        return {
            "n": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "max_ms": round(float(arr[-1]), 3),
        }


class Scorer:
    """Loads the trained artifacts once; scores one transaction at a time."""

    def __init__(
        self,
        booster: Any,
        calibrator: IsotonicCalibrator,
        stats: EntityStats,
        *,
        lo: float,
        hi: float,
    ) -> None:
        self.booster = booster
        self.calibrator = calibrator
        self.stats = stats
        self.lo = lo
        self.hi = hi
        self.latency = LatencyRecorder()

        self._names = feature_names()
        self._n = len(self._names)
        self._numeric_idx = {
            name: i for i, name in enumerate(self._names)
            if name in set(PASSTHROUGH_NUMERIC) | set(DERIVED_NUMERIC)
        }
        # Category -> code, precomputed. Doing this lookup per row per column
        # with `list.index` would cost more than the model call.
        self._cat_codes: dict[str, dict[str, int]] = {}
        self._cat_idx: dict[str, int] = {}
        for i, name in enumerate(self._names):
            if name in CATEGORICAL_COLS:
                self._cat_idx[name] = i
                vocab = stats.category_vocab.get(name, [])
                self._cat_codes[name] = {v: c for c, v in enumerate(vocab)}
        self._buf = np.empty((1, self._n), dtype=np.float64)

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifact_dir: Path | None = None) -> Scorer:
        import lightgbm as lgb

        d = artifact_dir or settings.artifact_dir
        model_path = d / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing -- run `make train` first")

        booster = lgb.Booster(model_file=str(model_path))
        calibrator = IsotonicCalibrator.load(d / "calibrator.json")
        stats = EntityStats.from_json(d / "entity_stats.json")

        lo, hi = settings.auto_approve_below, settings.auto_block_above
        bands_path = d / "thresholds.json"
        if bands_path.exists():
            chosen = json.loads(bands_path.read_text(encoding="utf-8"))["chosen"]
            lo, hi = float(chosen["lo"]), float(chosen["hi"])
        return cls(booster, calibrator, stats, lo=lo, hi=hi)

    # ------------------------------------------------------------------

    def _encode(self, row: Mapping[str, Any]) -> NDArray[np.float64]:
        buf = self._buf
        buf.fill(np.nan)
        for name, i in self._numeric_idx.items():
            v = row.get(name)
            if v is None:
                continue
            try:
                buf[0, i] = float(v)
            except (TypeError, ValueError):
                pass
        for name, i in self._cat_idx.items():
            code = self._cat_codes[name].get(_as_key(row.get(name)))
            # An unseen level stays NaN. LightGBM routes missing categoricals
            # down the default branch, which is the same thing the training
            # frame did for values outside the pinned vocabulary.
            if code is not None:
                buf[0, i] = float(code)
        return buf

    def score_row(self, row: Mapping[str, Any], transaction_id: str = "unknown") -> ScoreResult:
        t0 = time.perf_counter()
        arr = self._encode(row)
        raw = float(self.booster.predict(arr, raw_score=True)[0])
        prob = self.calibrator.transform_scalar(raw)
        band = self.band_for(prob)
        ms = round((time.perf_counter() - t0) * 1000.0, 4)
        self.latency.record(ms)
        return ScoreResult(
            transaction_id=transaction_id,
            score=prob,
            raw_score=raw,
            band=band,
            lo=self.lo,
            hi=self.hi,
            latency_ms=ms,
        )

    def band_for(self, prob: float) -> Band:
        if prob < self.lo:
            return Band.AUTO_APPROVE
        if prob >= self.hi:
            return Band.AUTO_BLOCK
        return Band.TRIAGE

    def score_frame(self, X: Any) -> NDArray[np.float64]:
        """Batch path, for evals and the threshold sweep. Not the SLA path."""
        raw = np.asarray(self.booster.predict(X, raw_score=True), dtype=np.float64)
        return self.calibrator.transform(raw)

    def explain_row(self, row: Mapping[str, Any], top_n: int = 8) -> list[dict[str, Any]]:
        """Per-transaction SHAP-style contributions from LightGBM's `pred_contrib`.

        The triage agent is given these as *evidence*, not as an instruction --
        the brief has to be traceable to something, and "the model weighted
        `card1_seen_count` heavily and this card is unseen" is a checkable
        claim. Costs ~2x a plain prediction, which is why it runs only on the
        triage band and not on every transaction.
        """
        arr = self._encode(row)
        contrib = np.asarray(self.booster.predict(arr, pred_contrib=True), dtype=np.float64)[0]
        # Last element is the expected-value bias term.
        pairs = [
            {"feature": name, "contribution": round(float(contrib[i]), 5), "value": row.get(name)}
            for i, name in enumerate(self._names)
        ]
        pairs.sort(key=lambda p: -abs(float(p["contribution"])))
        return [p for p in pairs[:top_n] if abs(float(p["contribution"])) > 1e-6]


def _as_key(value: Any) -> str:
    from fraudsight.features.spec import _norm_key

    return _norm_key(value)


def route_summary(results: list[ScoreResult]) -> dict[str, Any]:
    """Band distribution -- the volume funnel the demo narrates."""
    n = len(results) or 1
    counts = {b: 0 for b in Band}
    for r in results:
        counts[r.band] += 1
    return {
        "n": len(results),
        "counts": {str(k): v for k, v in counts.items()},
        "rates": {str(k): round(v / n, 6) for k, v in counts.items()},
    }
