# Postmortem — 16 tps, and the model was not the problem

- **Date:** 2026-08-15
- **Severity:** high (30× throughput deficit, plus a fabricated evidence signal)
- **Detected by:** a 40,000-transaction replay taking 42 minutes
- **Status:** resolved

---

## Summary

A replay of 40,000 transactions through the scoring path — no LLM calls at all —
sustained **16 transactions per second**. The NFR is 2,000.

The obvious suspect was the model. It was not: LightGBM plus calibration was
taking ~3.6 ms per transaction, comfortably inside budget. The remaining ~58 ms
was the **feature store**, computing sliding-window velocity counters.

Two independent bugs, one of which was also a correctness problem.

## Bug 1 — O(n²) window computation

`LocalFeatureStore.snapshot()` computed each window by scanning backwards
through the key's event deque and breaking at the first event outside the widest
window. That break is what made the algorithm look linear.

During a replay it never fires. Events are replayed in event-time order across a
30-day window, and the widest window is 7 days, so for a hot key most stored
events are *inside* the retention horizon and the scan walks nearly all of them.
One lookup is O(events in window); a replay does one lookup per transaction, so
the whole run is O(n²).

Measured directly: 20,000 observe+snapshot pairs on one key took **155.9
seconds**.

## Bug 2 — `__missing__` treated as an entity

`_norm_key()` maps a null value to the sentinel `__missing__`. The store keyed
on that sentinel like any other value, so **every transaction with a null
address shared one key**.

Two consequences:

1. **A fabricated signal.** `addr1` velocity read 1,926 transactions in the last
   hour — that is "transactions with a missing address", not "transactions on
   this address". It is exactly the kind of number the triage agent is designed
   to cite as a velocity spike, and it would have been cited confidently and
   grounded correctly, because the number was genuinely in the evidence.
2. **The performance cliff.** That single key accumulated tens of thousands of
   events, and bug 1 rescanned all of them on every lookup.

## Why it was not caught earlier

- **Unit tests used small windows.** `test_windows_are_measured_in_event_time`
  observes five events. Both bugs need thousands to show.
- **The eval suite never exercises velocity.** Eval cases run without a warmed
  feature store, so counters are empty and the code path is effectively dead
  offline. This is the same coverage gap that hid the groundedness bug in
  [the other postmortem from the same day](2026-08-15-groundedness-false-positives.md).
- **16 tps still finishes.** It is slow, not broken, and nothing failed. Without
  a throughput target written down, this ships.

## Fixes

**Incremental windows.** Running `(count, sum)` per window plus a head pointer
into the event list. `advance(now)` moves each pointer forward past expired
events, subtracting as it goes. Event time is non-decreasing during a replay, so
each pointer only ever moves forward and the amortised cost is O(1) per event.
Pointers are absolute ordinals, not list indices, so compaction can drop the
front of the list without invalidating them.

**`__missing__` is not an entity.** `observe()` ignores it and `snapshot()`
returns zeros with `seconds_since_previous: None`. "No data" instead of a
fabricated aggregate.

**A hard per-key event cap** (5,000) so no single key can degrade the store,
with the running totals corrected as capped events are dropped — dropping them
silently would have left that key permanently overstated.

## Results

| | Before | After |
|---|---|---|
| 20k observe+snapshot on one key | 155.9 s | **0.269 s** (580×) |
| 40,000-transaction replay | 16 tps | **487 tps** (30×) |
| Scoring p99 across the replay | 35.0 ms | **6.14 ms** |
| Integration suite | 517 s | **19 s** |
| `addr1` velocity on a null address | 1,926/hour | **0** |

## Tests added

- `test_incremental_windows_match_a_full_rescan` — 3,000 randomised steps
  against the naive implementation. An optimisation that quietly drifts is worse
  than the slow version it replaced.
- `test_missing_entity_has_no_velocity`
- `test_hot_key_stays_cheap` — 20,000 operations under 10 s
- `test_per_key_event_cap_keeps_totals_consistent`

## What we learned

**Profile before optimising, and profile the thing you did not suspect.** Every
design conversation about this system's latency was about the model. The model
was fine. The cost was in a utility class nobody had thought about since writing
it.

**A sentinel is not a value.** `__missing__` is a perfectly good marker for "we
do not know this" and a terrible key for "this entity". The bug produced a number
that was internally consistent, correctly grounded, and completely meaningless —
which is worse than a number that is obviously wrong.

**Small-N tests do not test algorithmic complexity.** All four velocity tests
passed throughout. Any test that asserts a time bound needs a size that makes the
bound meaningful.

## Follow-up

- [x] Regression test per fix
- [x] Throughput reported by `fraudsight replay` rather than assumed
- [ ] Warm the feature store in the eval runner (the shared coverage gap)
- [ ] Benchmark the Redis backend the same way — it has the same window
      semantics and has never been measured under load
