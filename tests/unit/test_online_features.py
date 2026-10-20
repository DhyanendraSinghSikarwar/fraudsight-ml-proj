"""Velocity counters: event-time windows and the off-by-one that inflates them."""

from __future__ import annotations

import pytest

import random

from fraudsight.features.spec import EntityStats
from fraudsight.features.store import (
    MAX_EVENTS_PER_KEY,
    MISSING_KEY,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    WINDOWS,
    LocalFeatureStore,
)


@pytest.fixture
def store() -> LocalFeatureStore:
    return LocalFeatureStore()


def test_windows_are_measured_in_event_time(store: LocalFeatureStore) -> None:
    """The replay harness compresses 30 days into minutes. A wall-clock window
    would put the whole replay in one bucket and report a velocity of thousands."""
    base = 10 * SECONDS_PER_DAY
    for i in range(5):
        store.observe("card1", "c1", event_time=base + i * 600, amount=10.0)

    snap = store.snapshot("card1", "c1", event_time=base + 4 * 600)
    assert snap.counts["1h"] == 5
    assert snap.amounts["1h"] == pytest.approx(50.0)


def test_events_outside_the_window_are_excluded(store: LocalFeatureStore) -> None:
    base = 10 * SECONDS_PER_DAY
    store.observe("card1", "c1", event_time=base, amount=10.0)
    store.observe("card1", "c1", event_time=base + 2 * SECONDS_PER_HOUR, amount=20.0)

    snap = store.snapshot("card1", "c1", event_time=base + 2 * SECONDS_PER_HOUR)
    assert snap.counts["1h"] == 1, "the 2-hour-old event must fall outside the 1h window"
    assert snap.counts["24h"] == 2


def test_seconds_since_previous_excludes_the_current_event(store: LocalFeatureStore) -> None:
    base = 10 * SECONDS_PER_DAY
    store.observe("card1", "c1", event_time=base, amount=10.0)
    snap = store.snapshot("card1", "c1", event_time=base + 900)
    assert snap.seconds_since_previous == pytest.approx(900.0)


def test_unknown_entity_returns_zeroes_not_an_error(store: LocalFeatureStore) -> None:
    snap = store.snapshot("card1", "never-seen", event_time=100_000)
    assert snap.counts["1h"] == 0
    assert snap.seconds_since_previous is None
    assert snap.as_features()["card1_seconds_since_prev"] == -1.0


def test_snapshot_excludes_current_txn() -> None:
    """The transaction being scored must not be counted in its own velocity.
    Counting it makes a card's first purchase look like a repeat and inflates
    every window by one."""
    stats = EntityStats(card1_counts={}, category_vocab={})
    from fraudsight.features.online import OnlineFeatureService

    svc = OnlineFeatureService(stats, LocalFeatureStore())
    base = 10 * SECONDS_PER_DAY
    txn = {"TransactionID": 1, "TransactionDT": base, "TransactionAmt": 50.0, "card1": 7}

    first = svc.compute(txn)
    assert first.snapshots["card1"].counts["1h"] == 0, "first ever transaction saw itself"

    second = svc.compute({**txn, "TransactionID": 2, "TransactionDT": base + 60})
    assert second.snapshots["card1"].counts["1h"] == 1


def test_warm_primes_windows_without_scoring() -> None:
    stats = EntityStats(card1_counts={}, category_vocab={})
    from fraudsight.features.online import OnlineFeatureService

    svc = OnlineFeatureService(stats, LocalFeatureStore())
    base = 10 * SECONDS_PER_DAY
    history = [
        {"TransactionID": i, "TransactionDT": base + i * 60, "TransactionAmt": 5.0, "card1": 7}
        for i in range(4)
    ]
    assert svc.warm(history) == 4

    live = svc.compute(
        {"TransactionID": 99, "TransactionDT": base + 500, "TransactionAmt": 5.0, "card1": 7}
    )
    assert live.snapshots["card1"].counts["1h"] == 4


def test_key_eviction_bounds_memory() -> None:
    store = LocalFeatureStore(max_keys=10)
    for i in range(50):
        store.observe("card1", f"c{i}", event_time=100_000 + i, amount=1.0)
    assert store.tracked_keys() <= 11


def _naive_windows(events: list[tuple[int, float]], now: int) -> tuple[dict, dict]:
    """The obvious O(n) rescan. The incremental implementation must match it."""
    counts = dict.fromkeys(WINDOWS, 0)
    amounts = {w: 0.0 for w in WINDOWS}
    for ts, amt in events:
        for w, span in WINDOWS.items():
            if now - ts <= span:
                counts[w] += 1
                amounts[w] += amt
    return counts, amounts


def test_incremental_windows_match_a_full_rescan(store: LocalFeatureStore) -> None:
    """The running totals are an optimisation over rescanning, and an
    optimisation that quietly drifts is worse than the slow version. 3,000
    randomised steps against the naive implementation."""
    rng = random.Random(7)
    base = 10 * SECONDS_PER_DAY
    events: list[tuple[int, float]] = []
    t = base

    for _ in range(3_000):
        t += rng.randint(1, 4_000)
        snap = store.snapshot("card1", "k", event_time=t)
        expected_counts, expected_amounts = _naive_windows(events, t)
        assert snap.counts == expected_counts, f"counts drifted at t={t}"
        for w in WINDOWS:
            assert snap.amounts[w] == pytest.approx(expected_amounts[w], abs=0.02)
        amount = round(rng.uniform(1, 500), 2)
        store.observe("card1", "k", event_time=t, amount=amount)
        events.append((t, amount))


def test_missing_entity_has_no_velocity(store: LocalFeatureStore) -> None:
    """Every transaction with a null address shares the `__missing__` sentinel.
    Aggregating them reports thousands of transactions per hour on "the missing
    address" -- a fabricated signal the agent would cite as a velocity spike."""
    base = 10 * SECONDS_PER_DAY
    for i in range(500):
        store.observe("addr1", MISSING_KEY, event_time=base + i, amount=10.0)

    snap = store.snapshot("addr1", MISSING_KEY, event_time=base + 500)
    assert snap.counts == dict.fromkeys(WINDOWS, 0)
    assert snap.seconds_since_previous is None
    assert store.tracked_keys() == 0, "a missing key must not consume a tracked slot"


def test_hot_key_stays_cheap(store: LocalFeatureStore) -> None:
    """Guards the O(n^2) regression: rescanning per lookup made a 40,000-row
    replay spend ~58ms per transaction in this store against ~3.6ms of scoring."""
    import time

    base = 10 * SECONDS_PER_DAY
    t0 = time.perf_counter()
    for i in range(20_000):
        store.observe("card1", "hot", event_time=base + i, amount=1.0)
        store.snapshot("card1", "hot", event_time=base + i)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"20k observe+snapshot took {elapsed:.1f}s -- the rescan is back"


def test_per_key_event_cap_keeps_totals_consistent(store: LocalFeatureStore) -> None:
    """Dropping capped events without correcting the running totals would leave
    a hot key permanently overstated."""
    base = 10 * SECONDS_PER_DAY
    n = MAX_EVENTS_PER_KEY + 2_000
    for i in range(n):
        store.observe("card1", "capped", event_time=base + i, amount=1.0)

    snap = store.snapshot("card1", "capped", event_time=base + n)
    assert snap.counts["7d"] <= MAX_EVENTS_PER_KEY
    # One event per second, so the last hour holds exactly 3,600 of them.
    assert snap.counts["1h"] == pytest.approx(3_600, abs=1)
    assert snap.amounts["1h"] == pytest.approx(float(snap.counts["1h"]), abs=0.01)


def test_velocity_is_not_a_model_feature() -> None:
    """The scorer trained on a flat table with no session context. Feeding it a
    velocity feature at serve time would be textbook train/serve skew."""
    from fraudsight.features.spec import feature_names

    names = set(feature_names())
    assert not {n for n in names if "_txn_1h" in n or "_seconds_since_prev" in n}
