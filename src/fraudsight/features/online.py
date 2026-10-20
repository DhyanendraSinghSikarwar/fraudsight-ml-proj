"""Real-time feature computation for one transaction.

The contract with the offline path is enforced, not hoped for:
`spec.row_features()` is the *same function* the training backfill validates
against, so there is exactly one implementation of "what does `amt_vs_card_p95`
mean". Velocity counters are layered on top and kept out of the model input --
see features/store.py for why.

Ordering matters and is easy to get wrong: the snapshot is taken **before** the
current transaction is observed. Counting the transaction you are scoring
inflates every velocity feature by one and makes a card's first-ever purchase
look like a repeat. The `observe_after=True` default encodes that; the test
tests/unit/test_online_features.py::test_snapshot_excludes_current_txn is what
keeps it encoded.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fraudsight.features.spec import (
    AMOUNT,
    TIME,
    TXN_ID,
    EntityStats,
    _norm_key,  # single normalisation rule, shared with the offline path
    row_features,
)
from fraudsight.features.store import ENTITIES, FeatureStore, VelocitySnapshot, build_store


@dataclass
class OnlineFeatures:
    """Everything the scoring and triage paths need from one transaction."""

    transaction_id: str
    event_time: int
    model_row: dict[str, Any]
    velocity: dict[str, float] = field(default_factory=dict)
    snapshots: dict[str, VelocitySnapshot] = field(default_factory=dict)
    compute_ms: float = 0.0

    def velocity_summary(self) -> dict[str, dict[str, Any]]:
        """Human/LLM-readable form. The raw flat dict is for metrics; this is
        what goes into an evidence bundle.

        `window_hours` and `window_days` look redundant next to the key names,
        and they are not. A brief that says "16 transactions in the preceding 24
        hours" cites two numbers: 16, which is a measurement, and 24, which is
        the size of the window it was measured over. The groundedness check has
        no way to tell those apart, so it flagged the window size as an invented
        figure on every velocity signal -- 59% of briefs in a replay. Window
        sizes are evidence; they belong in the evidence.
        """
        return {
            entity: {
                "key": snap.key,
                "txn_last_1h": snap.counts.get("1h", 0),
                "txn_last_24h": snap.counts.get("24h", 0),
                "txn_last_7d": snap.counts.get("7d", 0),
                "amount_last_24h": round(snap.amounts.get("24h", 0.0), 2),
                "seconds_since_previous": snap.seconds_since_previous,
                "window_hours": [1, 24, 168],
                "window_days": [7],
            }
            for entity, snap in self.snapshots.items()
        }


class OnlineFeatureService:
    def __init__(
        self,
        stats: EntityStats,
        store: FeatureStore | None = None,
        *,
        entities: tuple[str, ...] = ENTITIES,
    ) -> None:
        self.stats = stats
        self.store = store or build_store()
        self.entities = entities

    def compute(
        self, raw: Mapping[str, Any], *, observe_after: bool = True
    ) -> OnlineFeatures:
        t0 = time.perf_counter()
        event_time = int(float(raw.get(TIME) or 0))
        amount = float(raw.get(AMOUNT) or 0.0)

        snapshots: dict[str, VelocitySnapshot] = {}
        velocity: dict[str, float] = {}
        for entity in self.entities:
            key = _norm_key(raw.get(entity))
            snap = self.store.snapshot(entity, key, event_time=event_time)
            snapshots[entity] = snap
            velocity.update(snap.as_features())

        model_row = row_features(raw, self.stats)

        if observe_after:
            for entity in self.entities:
                self.store.observe(
                    entity, _norm_key(raw.get(entity)), event_time=event_time, amount=amount
                )

        return OnlineFeatures(
            transaction_id=str(raw.get(TXN_ID, "unknown")),
            event_time=event_time,
            model_row=model_row,
            velocity=velocity,
            snapshots=snapshots,
            compute_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    def warm(self, rows: list[Mapping[str, Any]]) -> int:
        """Prime the windows from history without scoring.

        The replay harness calls this so the first replayed transaction has a
        realistic velocity context instead of looking like a brand-new card.
        """
        n = 0
        for raw in rows:
            event_time = int(float(raw.get(TIME) or 0))
            amount = float(raw.get(AMOUNT) or 0.0)
            for entity in self.entities:
                self.store.observe(
                    entity, _norm_key(raw.get(entity)), event_time=event_time, amount=amount
                )
            n += 1
        return n
