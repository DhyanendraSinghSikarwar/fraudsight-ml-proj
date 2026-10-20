"""Split loading with pinned categorical dtypes.

Parquet stores each file's dictionary independently, so reading train.parquet
and val.parquet back gives two different string->code mappings for the same
column. LightGBM would then train on one encoding and predict on another --
silently, with no error and a plausible-looking score. Re-applying the
train-fitted vocabulary on load is what prevents that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraudsight.config import settings
from fraudsight.features.spec import (
    CATEGORICAL_COLS,
    LABEL,
    OTHER,
    TIME,
    TXN_ID,
    EntityStats,
    feature_names,
)


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: NDArray[np.int_]
    txn_id: NDArray[np.int64]
    time: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    def slice(self, mask: NDArray[np.bool_], name: str | None = None) -> Split:
        return Split(
            name=name or self.name,
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            txn_id=self.txn_id[mask],
            time=self.time[mask],
        )

    def split_by_time_median(self) -> tuple[Split, Split]:
        """Chronological halves. Used to keep early stopping and calibration on
        disjoint data -- calibrating on the same rows that chose the iteration
        count produces optimistically sharp probabilities."""
        cut = int(np.median(self.time))
        early = self.time < cut
        return self.slice(early, f"{self.name}_early"), self.slice(~early, f"{self.name}_late")


def load_stats() -> EntityStats:
    path = settings.artifact_dir / "entity_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make features` first")
    return EntityStats.from_json(path)


def load_split(name: str, stats: EntityStats | None = None) -> Split:
    stats = stats or load_stats()
    path = settings.data_dir / "processed" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make features` first")
    df = pd.read_parquet(path)

    y = df[LABEL].to_numpy(dtype=np.int_)
    txn_id = df[TXN_ID].to_numpy(dtype=np.int64)
    time = df[TIME].to_numpy(dtype=np.int64)

    X = df[feature_names()].copy()
    for col in CATEGORICAL_COLS:
        keys = X[col].astype("object")
        allowed = set(stats.category_vocab.get(col, []))
        if allowed:
            keys = keys.where(keys.isin(allowed), other=OTHER)
        X[col] = keys.astype(stats.dtype_for(col))
    return Split(name=name, X=X, y=y, txn_id=txn_id, time=time)


def processed_exists() -> bool:
    d = settings.data_dir / "processed"
    return all((d / f"{n}.parquet").exists() for n in ("train", "val", "test"))


def artifact(*parts: str) -> Path:
    return settings.artifact_dir.joinpath(*parts)
