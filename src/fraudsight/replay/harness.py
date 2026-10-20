"""Historical replay at a configurable rate.

This is how a throughput claim gets made without a production feed, and an
interviewer will trust a replay harness far more than a live toy stream --
because a replay is reproducible and a toy feed is a demo.

Two properties make the numbers mean something:

* **Event time drives the feature store.** Transactions carry their original
  `TransactionDT`, so velocity windows behave exactly as they would have at the
  original pace even when 30 days of history is replayed in four minutes. Using
  wall clock would put the whole replay in one bucket and report card velocities
  in the thousands.

* **The scoring path and the triage path are measured separately.** They have
  different SLAs (50ms p99 vs 6s p95) and different cost profiles, and a single
  blended latency number would hide both. `triage_sample` exists for the same
  reason: at a 12% triage rate, replaying 100k transactions means 12k LLM calls,
  and the throughput of the *scoring* path should be measurable without paying
  for that.

The rate limiter is a token-bucket with a monotonic clock. `asyncio.sleep` per
transaction would cap throughput at roughly 1/sleep-resolution -- around 60 tps
on Windows -- and report that as the system's limit rather than the scheduler's.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraudsight.config import Settings, settings as default_settings
from fraudsight.features.spec import LABEL, TIME, TXN_ID
from fraudsight.scoring.serve import Band
from fraudsight.triage.pipeline import Decision, TriagePipeline


@dataclass
class ReplayHarness:
    pipeline: TriagePipeline
    frame: pd.DataFrame
    cfg: Settings
    triage_sample: float = 1.0
    decisions: list[Decision] = field(default_factory=list)

    @classmethod
    async def build(
        cls,
        *,
        split: str = "test",
        cfg: Settings | None = None,
        triage_sample: float = 1.0,
        warm_rows: int = 20_000,
    ) -> ReplayHarness:
        cfg = cfg or default_settings
        pipeline = TriagePipeline.build(cfg)

        path = cfg.data_dir / "processed" / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `fraudsight features` first")
        frame = pd.read_parquet(path).sort_values(TIME, kind="stable").reset_index(drop=True)

        # Prime the velocity windows from the head of the split so the first
        # replayed transaction is not treated as every card's first ever.
        if warm_rows:
            head = frame.head(min(warm_rows, len(frame))).to_dict(orient="records")
            pipeline.features.warm(head)

        return cls(pipeline=pipeline, frame=frame, cfg=cfg, triage_sample=triage_sample)

    async def run(
        self,
        *,
        n: int = 2_000,
        target_rate: float = 0.0,
        concurrency: int = 16,
    ) -> dict[str, Any]:
        rows = self.frame.head(min(n, len(self.frame))).to_dict(orient="records")
        rng = np.random.default_rng(11)
        bucket = _TokenBucket(target_rate) if target_rate > 0 else None
        sem = asyncio.Semaphore(concurrency)

        started = time.perf_counter()
        score_only = 0
        triaged = 0
        skipped_triage = 0

        async def one(row: dict[str, Any]) -> None:
            nonlocal score_only, triaged, skipped_triage
            if bucket is not None:
                await bucket.acquire()

            feats = self.pipeline.features.compute(row)
            result = self.pipeline.scorer.score_row(feats.model_row, feats.transaction_id)

            if result.band is not Band.TRIAGE:
                score_only += 1
                self.decisions.append(
                    Decision(
                        transaction_id=feats.transaction_id,
                        score=result.score,
                        band=result.band,
                        action="approved" if result.band is Band.AUTO_APPROVE else "blocked",
                        score_ms=result.latency_ms,
                        feature_ms=feats.compute_ms,
                    )
                )
                return

            if rng.random() > self.triage_sample:
                skipped_triage += 1
                self.decisions.append(
                    Decision(
                        transaction_id=feats.transaction_id,
                        score=result.score,
                        band=result.band,
                        action="queued_for_triage_unsampled",
                        score_ms=result.latency_ms,
                        feature_ms=feats.compute_ms,
                    )
                )
                return

            async with sem:
                decision = await self.pipeline.process(row)
            triaged += 1
            self.decisions.append(decision)

        # Bounded batches, not one giant gather.
        #
        # `asyncio.gather` over the whole list creates every coroutine up front,
        # so a triage task scheduled at position 3,900 sits in the queue behind
        # thousands of scoring calls and its spans stretch to match. That
        # produced a p95 "triage latency" of 90 seconds on a path whose real
        # service time is ~1.4s -- a measurement of the harness, not the system.
        # A real service has bounded in-flight work; so does this.
        batch = max(concurrency * 4, 64)
        for start in range(0, len(rows), batch):
            await asyncio.gather(*(one(r) for r in rows[start : start + batch]))
        elapsed = time.perf_counter() - started

        return self._summary(rows, elapsed, score_only, triaged, skipped_triage)

    def _summary(
        self,
        rows: list[dict[str, Any]],
        elapsed: float,
        score_only: int,
        triaged: int,
        skipped: int,
    ) -> dict[str, Any]:
        labels = np.asarray([int(r.get(LABEL, 0)) for r in rows])
        window = self.decisions[-len(rows) :]

        # The funnel is computed from the harness's own decision list, not from
        # `pipeline.summary()`. Only transactions that reach `pipeline.process`
        # land in the pipeline's list, so reading the funnel from there reported
        # "100% triage" on a run that auto-approved 88% of its volume -- the
        # single most misleading number this harness could produce.
        by_band: dict[str, int] = {}
        for d in window:
            by_band[str(d.band)] = by_band.get(str(d.band), 0) + 1

        # Recall as the system defines it: fraud that was not auto-approved.
        approved_ids = {d.transaction_id for d in window if d.band is Band.AUTO_APPROVE}
        fraud_rows = [r for r in rows if int(r.get(LABEL, 0)) == 1]
        missed = sum(1 for r in fraud_rows if str(r.get(TXN_ID)) in approved_ids)
        recall = 1.0 - missed / len(fraud_rows) if fraud_rows else 0.0

        summary = self.pipeline.summary()
        n = len(rows) or 1
        cost = sum(d.cost_usd for d in window)

        # Service time, not wall clock. Every triage task is scheduled at the
        # start of the run and then competes with thousands of scoring calls on
        # one event loop, so a task's wall clock is dominated by how long it sat
        # in the queue -- that is a throughput observation, not a latency one.
        triage_service = [
            d.state.total_ms for d in window if d.state is not None and d.state.spans
        ]

        return {
            "transactions": len(rows),
            "elapsed_s": round(elapsed, 3),
            "achieved_tps": round(len(rows) / elapsed, 1) if elapsed else 0.0,
            # Throughput of the scoring path alone, which is the number the
            # 2,000 tps NFR is about. Derived from measured per-transaction
            # service time in one process -- see docs/scope.md for what
            # horizontal scaling would be required to reach the target.
            "scoring_path_tps_single_worker": round(
                1000.0 / max(summary["scoring_latency"].get("mean_ms", 1.0), 1e-6), 1
            ),
            "scored_only": score_only,
            "triaged": triaged,
            "triage_unsampled": skipped,
            "triage_sample": self.triage_sample,
            "fraud_in_window": int(labels.sum()),
            "fraud_auto_approved": missed,
            "recall": round(recall, 4),
            "band_counts": by_band,
            "band_rates": {k: round(v / n, 6) for k, v in by_band.items()},
            "scoring_latency": summary["scoring_latency"],
            "triage_latency_service_ms": _percentiles(triage_service),
            "cost_usd_total": round(cost, 6),
            "cost_per_triage_usd": round(cost / max(triaged, 1), 8),
            "cost_per_1k_transactions_usd": round(1000.0 * cost / n, 6),
            "provider_health": summary["provider_health"],
        }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr[-1]), 2),
    }


class _TokenBucket:
    """Rate limiting that does not itself become the bottleneck.

    Refills continuously against a monotonic clock and only sleeps when the
    bucket is genuinely empty, so the achieved rate tracks the target instead of
    the event loop's timer granularity.
    """

    def __init__(self, rate: float, burst: float = 64.0) -> None:
        self.rate = rate
        self.capacity = max(burst, rate * 0.1)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self.tokens) / self.rate)
