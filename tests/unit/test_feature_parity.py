"""The offline and online feature paths must agree, element for element.

This is the test that prevents train/serve skew, which is the failure mode that
does not raise, does not log, and shows up as a model that scores well offline
and badly in production. `row_features` is the reference implementation and
`frame_features` is its vectorised twin; if someone optimises one and not the
other, this fails.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraudsight.features.spec import (
    CATEGORICAL_COLS,
    DERIVED_NUMERIC,
    EntityStats,
    frame_features,
    row_features,
)
from tests.conftest import needs_artifacts


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TransactionID": 1, "TransactionDT": 86_400 * 3 + 3_600 * 2,
                "TransactionAmt": 100.0, "card1": 1000, "addr1": 315,
                "ProductCD": "W", "card4": "visa", "card6": "debit",
                "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
                "DeviceType": "desktop", "DeviceInfo": "Windows", "isFraud": 0,
                "C1": 1.0, "D1": 0.0, "id_01": -5.0,
            },
            {
                "TransactionID": 2, "TransactionDT": 86_400 * 4 + 60,
                "TransactionAmt": 5.0, "card1": 2000, "addr1": 204,
                "ProductCD": "C", "card4": "mastercard", "card6": "credit",
                "P_emaildomain": "yahoo.com", "R_emaildomain": None,
                "DeviceType": None, "DeviceInfo": None, "isFraud": 1,
                "C1": 9.0, "D1": 3.0, "id_01": None,
            },
            {
                "TransactionID": 3, "TransactionDT": 86_400 * 5 + 3_600 * 23,
                "TransactionAmt": 1999.99, "card1": 1000, "addr1": 315,
                "ProductCD": "W", "card4": "visa", "card6": "debit",
                "P_emaildomain": "gmail.com", "R_emaildomain": "hotmail.com",
                "DeviceType": "mobile", "DeviceInfo": "iOS", "isFraud": 0,
                "C1": 2.0, "D1": 1.0, "id_01": 0.0,
            },
        ]
    )


def test_row_and_frame_agree_on_derived_features() -> None:
    df = _toy_frame()
    stats = EntityStats.fit(df)
    framed = frame_features(df, stats)

    for i in range(len(df)):
        raw = df.iloc[i].to_dict()
        row = row_features(raw, stats)
        for col in DERIVED_NUMERIC:
            a, b = float(row[col]), float(framed.iloc[i][col])
            assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5), (
                f"row {i}, feature {col}: row_features={a} frame_features={b}"
            )


def test_row_and_frame_agree_on_categorical_levels() -> None:
    df = _toy_frame()
    stats = EntityStats.fit(df)
    framed = frame_features(df, stats)
    for i in range(len(df)):
        row = row_features(df.iloc[i].to_dict(), stats)
        for col in CATEGORICAL_COLS:
            assert str(row[col]) == str(framed.iloc[i][col]), f"row {i}, column {col}"


def test_missing_columns_do_not_crash_either_path() -> None:
    """Serving sees partial records. Both paths must degrade, not raise."""
    df = _toy_frame()
    stats = EntityStats.fit(df)
    sparse = {"TransactionID": 9, "TransactionDT": 100_000, "TransactionAmt": 42.0}
    row = row_features(sparse, stats)
    assert row["amt_log"] == pytest.approx(math.log1p(42.0))
    assert row["missing_field_count"] > 0

    framed = frame_features(pd.DataFrame([sparse]), stats)
    assert len(framed) == 1


def test_email_mismatch_is_three_valued() -> None:
    """match / mismatch / unknown are distinct. Collapsing unknown into
    mismatch invents a signal on every transaction with no recipient."""
    df = _toy_frame()
    stats = EntityStats.fit(df)
    same = row_features(
        {"P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com", "TransactionAmt": 1}, stats
    )
    diff = row_features(
        {"P_emaildomain": "gmail.com", "R_emaildomain": "yahoo.com", "TransactionAmt": 1}, stats
    )
    unknown = row_features({"P_emaildomain": "gmail.com", "TransactionAmt": 1}, stats)
    assert same["email_domains_match"] == 1.0
    assert diff["email_domains_match"] == 0.0
    assert unknown["email_domains_match"] == -1.0


@needs_artifacts
def test_parity_on_real_raw_rows() -> None:
    """The toy frame cannot exercise the real vocabulary or the V-block.

    Both paths are fed the same *raw* records. Feeding `row_features` a row from
    the processed parquet instead would compare a stored value against a
    recomputation from a different input -- `missing_field_count` in particular
    counts absent raw columns and cannot be recovered downstream -- and the
    mismatch would be the test's fault, not the code's.
    """
    from fraudsight.features.offline import load_raw
    from fraudsight.scoring.dataset import load_stats

    stats = load_stats()
    raw = load_raw(nrows=1_500)
    framed = frame_features(raw, stats)

    for i in range(0, len(raw), 137):
        record = raw.iloc[i].to_dict()
        row = row_features(record, stats)
        for col in DERIVED_NUMERIC:
            a, b = float(row[col]), float(framed.iloc[i][col])
            if np.isnan(a) and np.isnan(b):
                continue
            assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4), (
                f"row {i}, feature {col}: row_features={a} frame_features={b}"
            )
        for col in CATEGORICAL_COLS:
            assert str(row[col]) == str(framed.iloc[i][col]), f"row {i}, column {col}"
