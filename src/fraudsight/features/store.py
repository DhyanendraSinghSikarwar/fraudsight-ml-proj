"""Sliding-window velocity counters for the online path.

Everything here is keyed on **event time** (`TransactionDT`), never wall clock.
That single decision is what makes the replay harness meaningful: replaying 30
days of history in four minutes must produce the same counters it would have
produced at the original pace, and a wall-clock window would put the entire
replay inside one bucket and report a card velocity of 84,000/hour.

Two backends behind one interface:

* `local`  -- in-process deques. Exact, zero dependencies, bounded by a hard cap
              on tracked keys so a long replay cannot exhaust memory.
* `redis`  -- sorted sets with `ZREMRANGEBYSCORE` eviction, for more than one
              worker. Same semantics, same tests.

The counters are **not** model inputs. IEEE-CIS is a flat transaction table with
no session context, so a scorer trained on it never saw a velocity feature and
feeding it one at serve time would be textbook train/serve skew. They feed the
*evidence* layer instead -- the triage agent cites them, and the analyst reads
them. docs/scope.md records this and what it would take to change it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

# Windows the triage brief can talk about without a codebook.
WINDOWS: dict[str, int] = {
    "1h": SECONDS_PER_HOUR,
    "24h": SECONDS_PER_DAY,
    "7d": 7 * SECONDS_PER_DAY,
}

# Entities we track velocity for. Card and address are the classic card-testing
# and account-takeover signals; device catches a mule farm reusing hardware.
ENTITIES: tuple[str, ...] = ("card1", "addr1", "DeviceInfo", "P_emaildomain")

# Memory ceiling for the local backend. IEEE-CIS has ~17k distinct card1 values,
# so this is ~10x headroom; past it we evict the least-recently-touched key,
# which degrades a rare card's counters rather than the whole process.
MAX_TRACKED_KEYS = 200_000

# The sentinel `_norm_key` produces for a null entity. It is NOT an entity:
# every transaction with no address shares it, so "transactions on this address
# in the last hour" becomes "transactions with a missing address in the last
# hour" -- a number in the thousands that means nothing and that the triage
# agent would cite as a velocity spike.
#
# It is also a performance cliff. The window scan is O(events in window) per
# lookup, so one key holding every null-address transaction turns a replay into
# O(n^2): a 40,000-row replay spent ~58ms per transaction in this scan against
# ~3.6ms of actual scoring.
MISSING_KEY = "__missing__"

# Hard ceiling on events retained per key. Past this the oldest are dropped.
# A card with 5,000 transactions in a week is either a merchant aggregator or a
# data error; either way the marginal event adds nothing to a velocity signal
# and the scan cost is real.
MAX_EVENTS_PER_KEY = 5_000


@dataclass(frozen=True)
class VelocitySnapshot:
    """What one entity has done inside each window, as of an event time."""

    entity: str
    key: str
    counts: dict[str, int]
    amounts: dict[str, float]
    seconds_since_previous: float | None

    def as_features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for w in WINDOWS:
            out[f"{self.entity}_txn_{w}"] = float(self.counts.get(w, 0))
            out[f"{self.entity}_amt_{w}"] = round(float(self.amounts.get(w, 0.0)), 2)
        out[f"{self.entity}_seconds_since_prev"] = (
            -1.0 if self.seconds_since_previous is None else float(self.seconds_since_previous)
        )
        return out


class FeatureStore(Protocol):
    def observe(self, entity: str, key: str, *, event_time: int, amount: float) -> None: ...

    def snapshot(self, entity: str, key: str, *, event_time: int) -> VelocitySnapshot: ...

    def clear(self) -> None: ...


@dataclass
class _KeyWindows:
    """Incremental sliding windows for one entity key.

    Running totals with a head pointer per window, not a rescan. The rescan
    version was O(events in window) per lookup, which on a hot key is O(n^2)
    over a replay: a 40,000-row replay spent ~58 ms per transaction here
    against ~3.6 ms of actual model scoring. Because event time is
    non-decreasing during a replay, each pointer only ever moves forward, so
    the amortised cost is O(1) per event.

    The `heads` are absolute ordinals rather than list indices so that
    compaction can drop the front of `events` without invalidating them.
    """

    events: list[tuple[int, float]] = field(default_factory=list)
    dropped: int = 0  # events compacted off the front, for ordinal arithmetic
    heads: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WINDOWS, 0))
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WINDOWS, 0))
    amounts: dict[str, float] = field(default_factory=lambda: {w: 0.0 for w in WINDOWS})
    last_time: int | None = None

    def observe(self, event_time: int, amount: float) -> None:
        self.events.append((event_time, amount))
        for w in WINDOWS:
            self.counts[w] += 1
            self.amounts[w] += amount
        self.last_time = event_time

    def advance(self, now: int) -> None:
        """Expire events that have fallen out of each window."""
        for name, span in WINDOWS.items():
            head = self.heads[name]
            while head - self.dropped < len(self.events):
                ts, amt = self.events[head - self.dropped]
                if now - ts <= span:
                    break
                self.counts[name] -= 1
                self.amounts[name] -= amt
                head += 1
            self.heads[name] = head
        self._compact()

    def _compact(self) -> None:
        """Drop events every window has passed, then cap pathological keys."""
        # Everything below the earliest head has expired from every window and
        # contributes to no running total, so it can go without adjustment.
        floor = min(self.heads.values())
        cut = floor - self.dropped
        if cut > 0:
            del self.events[:cut]
            self.dropped = floor

        # Hard cap. These events are still inside at least one window, so each
        # window that still counts them has to be corrected as they go --
        # dropping them silently would leave the running totals permanently
        # overstated for this key.
        if len(self.events) <= MAX_EVENTS_PER_KEY:
            return
        excess = len(self.events) - MAX_EVENTS_PER_KEY
        for i in range(excess):
            ordinal = self.dropped + i
            ts, amt = self.events[i]
            for name in WINDOWS:
                if self.heads[name] <= ordinal:
                    self.counts[name] -= 1
                    self.amounts[name] -= amt
                    self.heads[name] = ordinal + 1
        del self.events[:excess]
        self.dropped += excess

    def previous_gap(self, now: int) -> float | None:
        if self.last_time is None or self.last_time >= now:
            return None
        return float(now - self.last_time)


class LocalFeatureStore:
    """In-process sliding windows. Exact within the retention horizon."""

    def __init__(self, max_keys: int = MAX_TRACKED_KEYS) -> None:
        self._keys: dict[tuple[str, str], _KeyWindows] = {}
        self._last_seen: dict[tuple[str, str], int] = {}
        self._max_keys = max_keys
        self._retention = max(WINDOWS.values())

    def observe(self, entity: str, key: str, *, event_time: int, amount: float) -> None:
        if key == MISSING_KEY:
            return
        k = (entity, key)
        state = self._keys.get(k)
        if state is None:
            state = self._keys[k] = _KeyWindows()
        state.advance(event_time)
        state.observe(event_time, amount)
        self._last_seen[k] = event_time
        if len(self._keys) > self._max_keys:
            self._drop_coldest()

    def snapshot(self, entity: str, key: str, *, event_time: int) -> VelocitySnapshot:
        if key == MISSING_KEY:
            # An absent entity has no velocity. Returning zeros rather than the
            # aggregate over every null-keyed transaction is the difference
            # between "no data" and a fabricated signal -- the aggregate reads
            # as thousands of transactions per hour on "the missing address".
            return VelocitySnapshot(
                entity=entity,
                key=key,
                counts=dict.fromkeys(WINDOWS, 0),
                amounts={w: 0.0 for w in WINDOWS},
                seconds_since_previous=None,
            )

        state = self._keys.get((entity, key))
        if state is None:
            return VelocitySnapshot(
                entity=entity,
                key=key,
                counts=dict.fromkeys(WINDOWS, 0),
                amounts={w: 0.0 for w in WINDOWS},
                seconds_since_previous=None,
            )

        state.advance(event_time)
        return VelocitySnapshot(
            entity=entity,
            key=key,
            counts=dict(state.counts),
            amounts={w: round(v, 2) for w, v in state.amounts.items()},
            seconds_since_previous=state.previous_gap(event_time),
        )

    def clear(self) -> None:
        self._keys.clear()
        self._last_seen.clear()

    def tracked_keys(self) -> int:
        return len(self._keys)

    def _drop_coldest(self) -> None:
        victim = min(self._last_seen, key=lambda k: self._last_seen[k])
        self._keys.pop(victim, None)
        self._last_seen.pop(victim, None)


class RedisFeatureStore:
    """Sorted-set backed windows, for more than one scoring worker.

    `member` carries the amount and a counter so identical (time, amount) pairs
    are distinct members rather than silently deduplicated by the set -- a card
    tested with ten identical $1.00 authorisations in the same second is exactly
    the pattern this store exists to surface.
    """

    def __init__(self, url: str, *, namespace: str = "fs") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise RuntimeError("redis not installed. `uv pip install -e '.[db]'`") from exc
        self._r: Any = redis.Redis.from_url(url, decode_responses=True)
        self._ns = namespace
        self._retention = max(WINDOWS.values())
        self._seq = 0

    def _key(self, entity: str, key: str) -> str:
        return f"{self._ns}:{entity}:{key}"

    def observe(self, entity: str, key: str, *, event_time: int, amount: float) -> None:
        rk = self._key(entity, key)
        self._seq += 1
        pipe = self._r.pipeline()
        pipe.zadd(rk, {f"{amount:.2f}:{self._seq}": event_time})
        pipe.zremrangebyscore(rk, "-inf", event_time - self._retention)
        # Expire idle keys so a one-off card does not live forever.
        pipe.expire(rk, self._retention * 2)
        pipe.execute()

    def snapshot(self, entity: str, key: str, *, event_time: int) -> VelocitySnapshot:
        rk = self._key(entity, key)
        counts: dict[str, int] = {}
        amounts: dict[str, float] = {}
        for name, span in WINDOWS.items():
            members = self._r.zrangebyscore(rk, event_time - span, event_time)
            counts[name] = len(members)
            amounts[name] = round(sum(float(m.split(":")[0]) for m in members), 2)
        prev = self._r.zrangebyscore(rk, "-inf", f"({event_time}", start=0, num=1, withscores=True)
        gap = float(event_time - prev[-1][1]) if prev else None
        return VelocitySnapshot(
            entity=entity, key=key, counts=counts, amounts=amounts, seconds_since_previous=gap
        )

    def clear(self) -> None:
        for k in self._r.scan_iter(f"{self._ns}:*"):
            self._r.delete(k)


def build_store(backend: str | None = None, *, redis_url: str | None = None) -> FeatureStore:
    from fraudsight.config import settings

    backend = backend or settings.feature_backend
    if backend == "redis":
        try:
            return RedisFeatureStore(redis_url or settings.redis_url)
        except Exception:  # noqa: BLE001 - a missing Redis must not stop scoring
            # Falling back rather than failing: velocity evidence is an
            # enrichment. Losing it degrades the brief; losing the scorer would
            # drop transactions. The degradation is reported by /health.
            return LocalFeatureStore()
    return LocalFeatureStore()
