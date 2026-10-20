"""Offline (training) feature pipeline.

Two things here are load-bearing and both are about not lying to yourself:

1.  The split is **time-ordered with an embargo**, never random. Fraud drifts,
    and a random split lets the model see the future. The embargo additionally
    models label maturity: chargebacks land 30-90 days after the transaction,
    so at training time you genuinely do not have labels for the most recent
    window. Skipping it is the single most common way a fraud model looks
    better offline than it can possibly be in production.

2.  `EntityStats` is fit on the training rows only, then applied unchanged to
    val and test. tests/unit/test_no_leakage.py asserts the fit never saw a
    row outside the train window.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudsight.config import settings
from fraudsight.features.spec import (
    ALL_RAW_COLS,
    AMOUNT,
    CATEGORICAL_COLS,
    LABEL,
    PASSTHROUGH_NUMERIC,
    TIME,
    TXN_ID,
    EntityStats,
    frame_features,
)

# Chargeback maturity window.
#
# The honest production number is ~45 days for a mixed card-present /
# card-not-present portfolio. IEEE-CIS only spans 182 days, and a 45-day
# embargo on top of a 60% training window leaves ~28k rows and ~1k positives
# per eval split -- small enough that run-to-run noise would swamp any real
# improvement. So the default here is 14 days, which preserves ~30-day val and
# test windows, and the shortfall is recorded in docs/scope.md rather than
# hidden. On a multi-year internal history, set embargo_days=45.
EMBARGO_DAYS = 14
SECONDS_PER_DAY = 86_400

TRAIN_FRACTION = 0.60  # remaining post-embargo time is split 50/50 val/test


@dataclass(frozen=True)
class SplitBounds:
    """Wall-clock (TransactionDT) boundaries of each split."""

    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    embargo_days: int

    def describe(self) -> str:
        d = SECONDS_PER_DAY
        return (
            f"train [.. {self.train_end / d:.1f}d)  "
            f"embargo {self.embargo_days}d  "
            f"val [{self.val_start / d:.1f}d .. {self.val_end / d:.1f}d)  "
            f"test [{self.test_start / d:.1f}d .. {self.test_end / d:.1f}d]"
        )


def raw_paths() -> tuple[Path, Path]:
    raw = settings.data_dir / "raw"
    return raw / "ieee_train_transaction.csv", raw / "ieee_train_identity.csv"


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Load and join the transaction + identity tables, downcast to float32.

    394 + 41 raw columns at float64 is ~1.9GB; the float32 dtype map plus the
    curated V-block subset brings a full load to well under 1GB.
    """
    txn_path, ident_path = raw_paths()
    if not txn_path.exists():
        raise FileNotFoundError(
            f"{txn_path} not found. Run `make data` (or "
            f"`python scripts/fetch_data.py --dataset ieee`) first."
        )

    txn_header = pd.read_csv(txn_path, nrows=0).columns
    ident_header = pd.read_csv(ident_path, nrows=0).columns

    want = set(ALL_RAW_COLS)
    txn_cols = [c for c in txn_header if c in want]
    ident_cols = [c for c in ident_header if c in want or c == TXN_ID]

    numeric = set(PASSTHROUGH_NUMERIC) | {TIME}
    txn_dtypes = {c: "float32" for c in txn_cols if c in numeric and c != TIME}
    txn_dtypes[TIME] = "int64"
    txn_dtypes[TXN_ID] = "int64"
    txn_dtypes[LABEL] = "int8"
    for c in txn_cols:
        if c in CATEGORICAL_COLS:
            txn_dtypes[c] = "object"

    txn = pd.read_csv(txn_path, usecols=txn_cols, dtype=txn_dtypes, nrows=nrows)

    ident_dtypes: dict[str, str] = {TXN_ID: "int64"}
    for c in ident_cols:
        if c == TXN_ID:
            continue
        ident_dtypes[c] = "float32" if c in numeric else "object"
    ident = pd.read_csv(ident_path, usecols=ident_cols, dtype=ident_dtypes)

    df = txn.merge(ident, on=TXN_ID, how="left")
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, embargo_days: int = EMBARGO_DAYS) -> tuple[pd.DataFrame, SplitBounds]:
    """Attach a `split` column by transaction time, with an embargo after train.

    Returns the frame with rows in the embargo window dropped -- they are
    unusable, and keeping them around invites accidental reuse.
    """
    t = df[TIME].to_numpy()
    t_min, t_max = int(t.min()), int(t.max())
    span = t_max - t_min
    embargo = embargo_days * SECONDS_PER_DAY

    if span <= embargo * 2:
        raise ValueError(
            f"data spans {span / SECONDS_PER_DAY:.1f} days but the embargo is "
            f"{embargo_days} days -- not enough history to split honestly"
        )

    train_end = t_min + int(span * TRAIN_FRACTION)
    val_start = train_end + embargo
    if val_start >= t_max:
        raise ValueError(
            f"embargo of {embargo_days}d consumes the entire post-train window "
            f"({(t_max - train_end) / SECONDS_PER_DAY:.1f}d available)"
        )
    val_end = val_start + (t_max - val_start) // 2
    bounds = SplitBounds(
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=val_end,
        test_end=t_max,
        embargo_days=embargo_days,
    )

    split = np.full(len(df), "embargo", dtype=object)
    split[t < train_end] = "train"
    split[(t >= val_start) & (t < val_end)] = "val"
    split[t >= val_end] = "test"

    out = df.copy()
    out["split"] = split
    dropped = int((out["split"] == "embargo").sum())
    out = out[out["split"] != "embargo"].reset_index(drop=True)
    out.attrs["embargoed_rows"] = dropped
    return out, bounds


def build(nrows: int | None = None, embargo_days: int = EMBARGO_DAYS) -> dict[str, object]:
    """Full offline pipeline. Writes parquet + entity stats, returns a summary."""
    settings.ensure_dirs()
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    raw = load_raw(nrows=nrows)
    df, bounds = time_split(raw, embargo_days=embargo_days)
    embargoed = int(df.attrs.get("embargoed_rows", 0))

    train_mask = df["split"] == "train"
    stats = EntityStats.fit(df.loc[train_mask])
    stats.to_json(settings.artifact_dir / "entity_stats.json")

    X = frame_features(df, stats)
    y = df[LABEL].astype("int8")

    summary: dict[str, object] = {
        "rows_total": int(len(df)),
        "rows_embargoed": embargoed,
        "features": int(X.shape[1]),
        "bounds": asdict(bounds),
        "bounds_human": bounds.describe(),
        "splits": {},
    }

    for name in ("train", "val", "test"):
        mask = (df["split"] == name).to_numpy()
        part = X.loc[mask].copy()
        part[LABEL] = y[mask].to_numpy()
        part[TXN_ID] = df.loc[mask, TXN_ID].to_numpy()
        part[TIME] = df.loc[mask, TIME].to_numpy()
        out_path = processed / f"{name}.parquet"
        part.to_parquet(out_path, index=False, compression="zstd")

        n = int(mask.sum())
        pos = int(y[mask].sum())
        summary["splits"][name] = {  # type: ignore[index]
            "rows": n,
            "fraud": pos,
            "fraud_rate": round(pos / n, 6) if n else 0.0,
            "amt_mean": round(float(df.loc[mask, AMOUNT].mean()), 2),
            "path": str(out_path.relative_to(settings.data_dir.parent)),
        }

    (settings.artifact_dir / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
