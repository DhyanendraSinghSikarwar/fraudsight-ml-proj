"""Leakage tests: the model and the evidence layer must not see the future.

Every one of these guards a failure that produces *better* offline numbers,
which is what makes them worth writing. A leak never announces itself; it just
makes the report look good.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraudsight.features.offline import EMBARGO_DAYS, SECONDS_PER_DAY, time_split
from fraudsight.features.spec import LABEL, TIME, EntityStats
from tests.conftest import needs_artifacts


def _frame(n: int = 3_000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            TIME: np.sort(rng.integers(86_400, 86_400 * 180, n)),
            "TransactionAmt": rng.gamma(2.0, 60.0, n),
            "card1": rng.integers(1000, 1100, n),
            "addr1": rng.integers(200, 400, n),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "DeviceInfo": rng.choice(["Windows", "iOS", None], n),
            LABEL: (rng.random(n) < 0.035).astype(int),
        }
    )


def test_splits_are_time_ordered_and_disjoint() -> None:
    df, bounds = time_split(_frame())
    train = df[df["split"] == "train"][TIME]
    val = df[df["split"] == "val"][TIME]
    test = df[df["split"] == "test"][TIME]
    assert train.max() < val.min(), "a training row is later than a validation row"
    assert val.max() < test.min(), "a validation row is later than a test row"
    assert bounds.train_end < bounds.val_start


def test_embargo_gap_is_actually_enforced() -> None:
    """Chargebacks land 30-90 days late. Without the gap the model trains on
    labels that would not have existed at decision time."""
    df, bounds = time_split(_frame(), embargo_days=EMBARGO_DAYS)
    gap_days = (bounds.val_start - bounds.train_end) / SECONDS_PER_DAY
    assert gap_days == pytest.approx(EMBARGO_DAYS, abs=0.01)

    train_max = df[df["split"] == "train"][TIME].max()
    val_min = df[df["split"] == "val"][TIME].min()
    assert (val_min - train_max) / SECONDS_PER_DAY >= EMBARGO_DAYS - 0.01


def test_embargoed_rows_are_dropped_not_relabelled() -> None:
    """Keeping them around invites accidental reuse."""
    df, _ = time_split(_frame())
    assert "embargo" not in set(df["split"])
    assert df.attrs["embargoed_rows"] > 0


def test_entity_stats_never_see_a_label() -> None:
    """Frequency encodings are counts, not target encodings. If a label ever
    reaches these tables, the features become a leak with no nested-CV to save
    them."""
    df, _ = time_split(_frame())
    train = df[df["split"] == "train"]

    stats = EntityStats.fit(train)
    flipped = train.copy()
    flipped[LABEL] = 1 - flipped[LABEL]
    stats_flipped = EntityStats.fit(flipped)

    assert json.dumps(stats.__dict__, sort_keys=True) == json.dumps(
        stats_flipped.__dict__, sort_keys=True
    ), "EntityStats changed when only the labels changed -- it is reading the target"


def test_entity_stats_are_fit_on_train_rows_only() -> None:
    df, bounds = time_split(_frame())
    stats = EntityStats.fit(df[df["split"] == "train"])
    assert stats.fit_time_max < bounds.val_start
    assert stats.n_train_rows == int((df["split"] == "train").sum())


def test_too_short_a_history_raises_rather_than_shrinking_the_embargo() -> None:
    short = _frame(n=500)
    short[TIME] = np.sort(np.random.default_rng(1).integers(86_400, 86_400 * 20, 500))
    with pytest.raises(ValueError, match="not enough history"):
        time_split(short, embargo_days=14)


@needs_artifacts
def test_case_corpus_is_drawn_from_the_training_window_only() -> None:
    """A triage case that could retrieve neighbours from its own window would
    leak, and would inflate every Track B metric invisibly."""
    from fraudsight.config import settings
    from fraudsight.tools.vectorstore import LocalVectorStore

    store = LocalVectorStore.load(settings.artifact_dir / "cases")
    summary = json.loads(
        (settings.artifact_dir / "feature_summary.json").read_text(encoding="utf-8")
    )
    train_end_day = summary["bounds"]["train_end"] / SECONDS_PER_DAY

    days = [c.metadata.get("day", 0.0) for c in store.cases]
    assert max(days) <= train_end_day + 0.01, (
        f"corpus contains a case from day {max(days):.1f}, past the train window "
        f"ending at day {train_end_day:.1f}"
    )


@needs_artifacts
def test_profiles_are_built_from_the_training_window_only() -> None:
    from fraudsight.config import settings
    from fraudsight.data.profiles import ProfileTables

    tables = ProfileTables.load(settings.artifact_dir / "profiles")
    summary = json.loads(
        (settings.artifact_dir / "feature_summary.json").read_text(encoding="utf-8")
    )
    train_end_day = summary["bounds"]["train_end"] / SECONDS_PER_DAY
    assert tables.built_from["window_days"][1] <= train_end_day + 0.01


@needs_artifacts
def test_prior_fraud_is_evidence_only_and_never_a_model_feature() -> None:
    """Label-derived context is legitimate for an analyst and would be a leak in
    the scorer. The separation is the point."""
    from fraudsight.features.spec import feature_names

    names = set(feature_names())
    for leaked in ("prior_confirmed_fraud", "prior_fraud_rate", "fraud_rate_vs_portfolio"):
        assert leaked not in names, f"{leaked} reached the model input vector"
