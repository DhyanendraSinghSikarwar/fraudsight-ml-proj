"""Feature contract shared by the offline (training) and online (serving) paths.

Train/serve skew is the classic way an ML system quietly rots, so the two paths
are not allowed to drift here: `row_features()` is the *reference*
implementation, `frame_features()` is its vectorised twin, and
tests/unit/test_feature_parity.py asserts they agree element-wise on a real
sample. If someone edits one and not the other, CI fails.

Everything learned from data (frequency maps, per-card amount baselines) lives
in `EntityStats`, which is fit on the training split ONLY and serialised to
artifacts/entity_stats.json. Nothing in here may look at a label.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column groups from the raw IEEE-CIS schema.
# --------------------------------------------------------------------------

LABEL = "isFraud"
TXN_ID = "TransactionID"
TIME = "TransactionDT"
AMOUNT = "TransactionAmt"

# Anonymised-but-informative numeric blocks the vendor ships.
#   C* = counting features (how many addresses/cards share an attribute)
#   D* = timedeltas since a previous related event
#   M* = match flags (name on card matches, address matches, ...)
#   V* = a large engineered block; heavily collinear, so we keep a subset.
COUNT_COLS = tuple(f"C{i}" for i in range(1, 15))
DELTA_COLS = tuple(f"D{i}" for i in range(1, 16))
MATCH_COLS = tuple(f"M{i}" for i in range(1, 10))

# V-block subset. The full 339 columns cost ~4x the memory and training time
# for ~0.003 AUC-PR; docs/scope.md records the measurement. Chosen to span the
# distinct correlation clusters rather than the first N.
V_COLS = (
    "V12", "V13", "V19", "V20", "V29", "V30", "V35", "V36", "V37", "V38",
    "V44", "V45", "V47", "V48", "V53", "V54", "V55", "V56", "V61", "V62",
    "V67", "V70", "V75", "V76", "V78", "V80", "V82", "V83", "V86", "V87",
    "V94", "V96", "V97", "V99", "V126", "V127", "V128", "V130", "V131",
    "V143", "V149", "V152", "V156", "V158", "V160", "V165", "V166", "V169",
    "V170", "V173", "V187", "V189", "V201", "V203", "V204", "V207", "V208",
    "V210", "V212", "V215", "V218", "V220", "V221", "V222", "V234", "V257",
    "V258", "V261", "V264", "V266", "V267", "V271", "V274", "V277", "V281",
    "V283", "V285", "V286", "V289", "V291", "V294", "V296", "V307", "V308",
    "V310", "V312", "V313", "V314", "V315", "V317", "V322", "V323", "V324",
)

CARD_COLS = ("card1", "card2", "card3", "card5")
ADDR_COLS = ("addr1", "addr2")
DIST_COLS = ("dist1", "dist2")

# Identity-table columns worth carrying: device + network signals.
ID_NUMERIC = tuple(f"id_{i:02d}" for i in range(1, 12))
ID_CATEGORICAL = ("id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31",
                  "id_33", "id_34", "id_35", "id_36", "id_37", "id_38")

CATEGORICAL_COLS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    *MATCH_COLS,
    *ID_CATEGORICAL,
)

PASSTHROUGH_NUMERIC = (
    AMOUNT,
    *CARD_COLS,
    *ADDR_COLS,
    *DIST_COLS,
    *COUNT_COLS,
    *DELTA_COLS,
    *V_COLS,
    *ID_NUMERIC,
)

# Features this module derives. Named so an analyst -- and the LLM writing the
# brief -- can read them without a codebook.
DERIVED_NUMERIC = (
    "hour_of_day",
    "day_of_week",
    "is_night",
    "amt_log",
    "amt_cents",
    "amt_is_round",
    "amt_vs_card_mean",
    "amt_vs_card_p95",
    "card1_seen_count",
    "addr1_seen_count",
    "email_seen_count",
    "device_seen_count",
    "email_domains_match",
    "has_identity_record",
    "missing_field_count",
)

ALL_RAW_COLS = (TXN_ID, LABEL, TIME, *PASSTHROUGH_NUMERIC, *CATEGORICAL_COLS)

SECONDS_PER_DAY = 86_400
# The vendor does not publish an epoch. TransactionDT starts at 86_400, and the
# hour-of-day histogram has its trough at DT % 86400 == 0, so treating the
# origin as local midnight lines the derived hour up with real diurnal fraud
# patterns. Documented rather than silently assumed.
DAY_ORIGIN_OFFSET = 0


def feature_names() -> list[str]:
    """Model input order. Stable -- the serving path depends on it."""
    return [*PASSTHROUGH_NUMERIC, *DERIVED_NUMERIC, *CATEGORICAL_COLS]


# --------------------------------------------------------------------------
# Learned statistics (fit on train split only)
# --------------------------------------------------------------------------


OTHER = "__other__"
MISSING = "__missing__"

# Cardinality cap per categorical column. DeviceInfo alone has ~2,800 distinct
# values in train, most appearing once; an unbounded vocabulary bloats the model
# and every rare level is noise. Values outside the top-K collapse to __other__.
MAX_CATEGORY_LEVELS = 200


@dataclass
class EntityStats:
    """Frequency and baseline statistics learned from the training split.

    Frequency encodings are counts of how often an entity appeared in training.
    They are *not* target encodings -- no label is touched -- which is what
    keeps this leak-free without a nested-CV scheme.
    """

    card1_counts: dict[str, int] = field(default_factory=dict)
    addr1_counts: dict[str, int] = field(default_factory=dict)
    email_counts: dict[str, int] = field(default_factory=dict)
    device_counts: dict[str, int] = field(default_factory=dict)
    card1_amt_mean: dict[str, float] = field(default_factory=dict)
    card1_amt_p95: dict[str, float] = field(default_factory=dict)
    # Pinned categorical vocabulary. Without this, each split's parquet carries
    # its own dictionary and LightGBM silently reads different integer codes for
    # the same string across train/val/test/serving.
    category_vocab: dict[str, list[str]] = field(default_factory=dict)
    global_amt_mean: float = 100.0
    global_amt_p95: float = 400.0
    n_train_rows: int = 0
    fit_time_max: int = 0  # max TransactionDT seen in training -- an embargo audit trail

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> EntityStats:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def fit(cls, train: pd.DataFrame) -> EntityStats:
        amt = train[AMOUNT].astype("float64")
        card_key = _key_series(train, "card1")
        grouped = amt.groupby(card_key)

        vocab: dict[str, list[str]] = {}
        for col in CATEGORICAL_COLS:
            keys = _key_series(train, col)
            top = keys.value_counts().head(MAX_CATEGORY_LEVELS).index.tolist()
            levels = sorted({str(k) for k in top} | {OTHER, MISSING})
            vocab[col] = levels

        return cls(
            card1_counts=_counts(card_key),
            addr1_counts=_counts(_key_series(train, "addr1")),
            email_counts=_counts(_key_series(train, "P_emaildomain")),
            device_counts=_counts(_key_series(train, "DeviceInfo")),
            card1_amt_mean={k: float(v) for k, v in grouped.mean().items()},
            card1_amt_p95={k: float(v) for k, v in grouped.quantile(0.95).items()},
            category_vocab=vocab,
            global_amt_mean=float(amt.mean()),
            global_amt_p95=float(amt.quantile(0.95)),
            n_train_rows=len(train),
            fit_time_max=int(train[TIME].max()),
        )

    def dtype_for(self, col: str) -> pd.CategoricalDtype:
        levels = self.category_vocab.get(col) or [OTHER, MISSING]
        return pd.CategoricalDtype(categories=levels, ordered=False)

    def coerce_level(self, col: str, key: str) -> str:
        """Map a normalised key onto the pinned vocabulary."""
        levels = self.category_vocab.get(col)
        if not levels:
            return key
        return key if key in levels else OTHER


def _key_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(["__missing__"] * len(df), index=df.index, dtype="object")
    return df[col].map(_norm_key)


def _counts(keys: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in keys.value_counts().items()}


def _norm_key(value: Any) -> str:
    """Canonical string key. NaN/None collapse to a single sentinel so an
    unseen entity and a missing entity are distinguishable downstream."""
    if value is None:
        return "__missing__"
    if isinstance(value, float):
        if math.isnan(value):
            return "__missing__"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    text = str(value).strip()
    return text.lower() if text and text.lower() != "nan" else "__missing__"


# --------------------------------------------------------------------------
# Reference implementation: one row at a time (this is what serving uses)
# --------------------------------------------------------------------------


def row_features(
    raw: Mapping[str, Any],
    stats: EntityStats,
    velocity: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Derive features for a single transaction.

    `velocity` carries live counters from the feature store (e.g. how many
    transactions this card made in the last hour). It is optional so the same
    function serves both the historical backfill and the real-time path.
    """
    dt = _as_float(raw.get(TIME), 0.0)
    amt = _as_float(raw.get(AMOUNT), 0.0)

    seconds_into_day = (int(dt) + DAY_ORIGIN_OFFSET) % SECONDS_PER_DAY
    hour = seconds_into_day // 3600
    day_index = (int(dt) + DAY_ORIGIN_OFFSET) // SECONDS_PER_DAY

    card_key = _norm_key(raw.get("card1"))
    card_mean = stats.card1_amt_mean.get(card_key, stats.global_amt_mean)
    card_p95 = stats.card1_amt_p95.get(card_key, stats.global_amt_p95)

    p_email = _norm_key(raw.get("P_emaildomain"))
    r_email = _norm_key(raw.get("R_emaildomain"))

    # An identity record exists iff the identity join produced any non-null.
    has_identity = any(
        raw.get(c) is not None and not _is_nan(raw.get(c))
        for c in ("DeviceType", "DeviceInfo", "id_01", "id_02")
    )

    tracked = (*PASSTHROUGH_NUMERIC, *CATEGORICAL_COLS)
    missing = sum(1 for c in tracked if raw.get(c) is None or _is_nan(raw.get(c)))

    cents = round((amt - math.floor(amt)) * 100)

    derived: dict[str, Any] = {
        "hour_of_day": float(hour),
        "day_of_week": float(day_index % 7),
        # 00:00-05:59 local. Card-not-present fraud concentrates overnight when
        # the genuine cardholder is least likely to notice a notification.
        "is_night": 1.0 if hour < 6 else 0.0,
        "amt_log": math.log1p(max(amt, 0.0)),
        "amt_cents": float(cents),
        # Round amounts are a weak but real card-testing signal.
        "amt_is_round": 1.0 if cents == 0 else 0.0,
        "amt_vs_card_mean": amt / card_mean if card_mean > 0 else 1.0,
        "amt_vs_card_p95": amt / card_p95 if card_p95 > 0 else 1.0,
        "card1_seen_count": float(stats.card1_counts.get(card_key, 0)),
        "addr1_seen_count": float(stats.addr1_counts.get(_norm_key(raw.get("addr1")), 0)),
        "email_seen_count": float(stats.email_counts.get(p_email, 0)),
        "device_seen_count": float(stats.device_counts.get(_norm_key(raw.get("DeviceInfo")), 0)),
        # Purchaser/recipient domain mismatch on a shipped good. Only meaningful
        # when both are present, hence the explicit missing branch.
        "email_domains_match": (
            1.0
            if p_email != "__missing__" and p_email == r_email
            else 0.0
            if r_email != "__missing__"
            else -1.0
        ),
        "has_identity_record": 1.0 if has_identity else 0.0,
        "missing_field_count": float(missing),
    }

    out: dict[str, Any] = {}
    for col in PASSTHROUGH_NUMERIC:
        out[col] = _as_float(raw.get(col), math.nan)
    out.update(derived)
    for col in CATEGORICAL_COLS:
        out[col] = stats.coerce_level(col, _norm_key(raw.get(col)))
    if velocity:
        out.update({k: float(v) for k, v in velocity.items()})
    return out


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) and math.isnan(default) is False else f


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


# --------------------------------------------------------------------------
# Vectorised twin: same math, whole frame at a time
# --------------------------------------------------------------------------


def frame_features(df: pd.DataFrame, stats: EntityStats) -> pd.DataFrame:
    """Vectorised equivalent of `row_features` for training-scale frames."""
    n = len(df)
    dt = df[TIME].astype("int64")
    amt = df[AMOUNT].astype("float64").fillna(0.0)

    seconds_into_day = (dt + DAY_ORIGIN_OFFSET) % SECONDS_PER_DAY
    hour = seconds_into_day // 3600
    day_index = (dt + DAY_ORIGIN_OFFSET) // SECONDS_PER_DAY

    card_key = _key_series(df, "card1")
    card_mean = card_key.map(stats.card1_amt_mean).fillna(stats.global_amt_mean)
    card_p95 = card_key.map(stats.card1_amt_p95).fillna(stats.global_amt_p95)

    p_email = _key_series(df, "P_emaildomain")
    r_email = _key_series(df, "R_emaildomain")

    identity_probe = [c for c in ("DeviceType", "DeviceInfo", "id_01", "id_02") if c in df.columns]
    has_identity = (
        df[identity_probe].notna().any(axis=1)
        if identity_probe
        else pd.Series(np.zeros(n, dtype=bool), index=df.index)
    )

    tracked = [c for c in (*PASSTHROUGH_NUMERIC, *CATEGORICAL_COLS) if c in df.columns]
    absent = len((*PASSTHROUGH_NUMERIC, *CATEGORICAL_COLS)) - len(tracked)
    missing = df[tracked].isna().sum(axis=1) + absent

    cents = ((amt - np.floor(amt)) * 100).round()

    # Assemble column-wise into a dict and concat once. Assigning ~190 columns
    # onto a DataFrame in a loop re-blocks the frame on every insert.
    cols: dict[str, pd.Series] = {}
    nan_col = pd.Series(np.full(n, np.nan, dtype="float32"), index=df.index)
    for col in PASSTHROUGH_NUMERIC:
        cols[col] = (
            pd.to_numeric(df[col], errors="coerce").astype("float32")
            if col in df.columns
            else nan_col
        )

    cols["hour_of_day"] = hour.astype("float32")
    cols["day_of_week"] = (day_index % 7).astype("float32")
    cols["is_night"] = (hour < 6).astype("float32")
    cols["amt_log"] = np.log1p(amt.clip(lower=0.0)).astype("float32")
    cols["amt_cents"] = cents.astype("float32")
    cols["amt_is_round"] = (cents == 0).astype("float32")
    cols["amt_vs_card_mean"] = _safe_div(amt, card_mean).astype("float32")
    cols["amt_vs_card_p95"] = _safe_div(amt, card_p95).astype("float32")
    cols["card1_seen_count"] = card_key.map(stats.card1_counts).fillna(0).astype("float32")
    cols["addr1_seen_count"] = (
        _key_series(df, "addr1").map(stats.addr1_counts).fillna(0).astype("float32")
    )
    cols["email_seen_count"] = p_email.map(stats.email_counts).fillna(0).astype("float32")
    cols["device_seen_count"] = (
        _key_series(df, "DeviceInfo").map(stats.device_counts).fillna(0).astype("float32")
    )
    cols["email_domains_match"] = pd.Series(
        np.where(
            (p_email != "__missing__") & (p_email == r_email),
            1.0,
            np.where(r_email != "__missing__", 0.0, -1.0),
        ).astype("float32"),
        index=df.index,
    )
    cols["has_identity_record"] = has_identity.astype("float32")
    cols["missing_field_count"] = missing.astype("float32")

    for col in CATEGORICAL_COLS:
        keys = _key_series(df, col)
        levels = stats.category_vocab.get(col)
        if levels:
            allowed = set(levels)
            keys = keys.where(keys.isin(allowed), other=OTHER)
        cols[col] = keys.astype(stats.dtype_for(col))

    return pd.concat(cols, axis=1)[feature_names()]


def frame_from_rows(rows: list[Mapping[str, Any]], stats: EntityStats) -> pd.DataFrame:
    """Build a model-ready frame from already-derived row dicts.

    Used by the serving path: `row_features` produces plain Python values, and
    this pins the categorical dtypes so the integer codes match training.
    """
    df = pd.DataFrame(list(rows))
    for col in feature_names():
        if col not in df.columns:
            df[col] = np.nan
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("object").fillna(MISSING).astype(stats.dtype_for(col))
    for col in (*PASSTHROUGH_NUMERIC, *DERIVED_NUMERIC):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df[feature_names()]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den > 0, other=np.nan)
    return (num / den).fillna(1.0)
