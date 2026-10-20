"""Spend accounting and the denial-of-wallet guard.

Two distinct jobs that are easy to conflate:

* **Accounting** -- every `Usage` produced anywhere in the system is recorded
  here, so `cost_per_triage` in the daily report is a *measured* number from
  real token counts rather than the assumed `settings.triage_llm_usd` constant
  that the cost matrix uses for band selection. When those two diverge, the band
  choice is stale and the report says so. That divergence is the thing a cost
  dashboard exists to catch.

* **Enforcement** -- a hard daily cap. An agent in a retry loop against a
  provider returning malformed JSON is a bill, not an outage, and it is the
  failure mode nobody instruments until it has happened once. `reserve()` is
  checked *before* the call rather than after, because a limit you discover by
  exceeding it is not a limit.

The ledger is deliberately process-local and append-only on disk. A distributed
counter would need Redis, and the honest scope of this system is one worker; the
file is what makes yesterday's spend readable by the daily report tomorrow.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fraudsight.llm.base import BudgetExceeded, Usage


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@dataclass
class DailyTotals:
    day: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    degraded_calls: int = 0
    refused_calls: int = 0
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_operation: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def as_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "degraded_calls": self.degraded_calls,
            "refused_calls": self.refused_calls,
            # Cache hit rate is the single most actionable cost lever on a path
            # with a static system prompt, so it is computed here rather than
            # left for a dashboard query to get wrong.
            "cache_hit_rate": round(
                self.cache_read_tokens
                / max(self.cache_read_tokens + self.input_tokens, 1),
                4,
            ),
            "by_model": {k: round(v, 6) for k, v in sorted(self.by_model.items())},
            "by_operation": {k: round(v, 6) for k, v in sorted(self.by_operation.items())},
        }


class CostLedger:
    """Thread-safe daily spend counter with a hard cap and a warning threshold."""

    def __init__(
        self,
        *,
        daily_budget_usd: float,
        alert_fraction: float = 0.8,
        ledger_path: Path | None = None,
    ) -> None:
        self.daily_budget_usd = daily_budget_usd
        self.alert_fraction = alert_fraction
        self._path = ledger_path
        self._lock = threading.Lock()
        self._days: dict[str, DailyTotals] = {}
        # Reservations held by calls that are in flight. Without this, N
        # concurrent triages each see the same pre-call total and collectively
        # blow through the cap; the estimate is released and replaced by the
        # measured cost on record().
        self._in_flight_usd = 0.0
        self._alerted_days: set[str] = set()

    # -- accounting ----------------------------------------------------

    def totals(self, day: str | None = None) -> DailyTotals:
        d = day or _today()
        with self._lock:
            return self._days.setdefault(d, DailyTotals(day=d))

    def spent_today(self) -> float:
        return self.totals().cost_usd

    def remaining_today(self) -> float:
        return max(0.0, self.daily_budget_usd - self.spent_today() - self._in_flight_usd)

    def record(self, usage: Usage, *, operation: str = "unknown", reserved: float = 0.0) -> Usage:
        day = _today()
        with self._lock:
            t = self._days.setdefault(day, DailyTotals(day=day))
            t.calls += 1
            t.input_tokens += usage.input_tokens
            t.output_tokens += usage.output_tokens
            t.cache_read_tokens += usage.cache_read_tokens
            t.cache_write_tokens += usage.cache_write_tokens
            t.cost_usd += usage.cost_usd
            t.degraded_calls += int(usage.degraded)
            t.refused_calls += int(usage.refused)
            t.by_model[f"{usage.provider}/{usage.model}"] += usage.cost_usd
            t.by_operation[operation] += usage.cost_usd
            self._in_flight_usd = max(0.0, self._in_flight_usd - reserved)
            over = t.cost_usd >= self.daily_budget_usd * self.alert_fraction
            first_alert = over and day not in self._alerted_days
            if first_alert:
                self._alerted_days.add(day)
        if self._path is not None:
            self._append(usage, operation=operation)
        if first_alert:
            self._warn(day)
        return usage

    def _warn(self, day: str) -> None:
        t = self.totals(day)
        pct = 100.0 * t.cost_usd / max(self.daily_budget_usd, 1e-9)
        # A print rather than a raise: crossing 80% is information, not an
        # incident. The hard stop lives in reserve().
        print(
            f"[fraudsight.cost] {day}: ${t.cost_usd:.4f} of ${self.daily_budget_usd:.2f} "
            f"({pct:.0f}% of daily budget) across {t.calls} calls",
            flush=True,
        )

    def _append(self, usage: Usage, *, operation: str) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "operation": operation,
            **usage.model_dump(),
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # -- enforcement ---------------------------------------------------

    def reserve(self, estimate_usd: float) -> float:
        """Hold budget for a call about to be made. Raises if it would breach.

        Returns the reserved amount so `record()` can release it.
        """
        with self._lock:
            day = _today()
            t = self._days.setdefault(day, DailyTotals(day=day))
            projected = t.cost_usd + self._in_flight_usd + estimate_usd
            if projected > self.daily_budget_usd:
                raise BudgetExceeded(
                    f"daily LLM budget ${self.daily_budget_usd:.2f} would be exceeded "
                    f"(spent ${t.cost_usd:.4f}, in flight ${self._in_flight_usd:.4f}, "
                    f"this call ~${estimate_usd:.4f}). Triage falls back to score-only "
                    f"routing; no transaction is dropped."
                )
            self._in_flight_usd += estimate_usd
            return estimate_usd

    def release(self, reserved: float) -> None:
        """Return a reservation when the call failed before producing usage."""
        with self._lock:
            self._in_flight_usd = max(0.0, self._in_flight_usd - reserved)

    def reset(self) -> None:
        with self._lock:
            self._days.clear()
            self._alerted_days.clear()
            self._in_flight_usd = 0.0


def estimate_cost(provider: str, model: str, *, system: str, user: str, max_tokens: int) -> float:
    """Pre-call cost estimate for the budget reservation.

    Deliberately pessimistic: input tokens from a chars/4 approximation, output
    assumed to hit `max_tokens`. Under-estimating here is what lets a budget cap
    be exceeded by the very call that was supposed to be checked against it.
    """
    from fraudsight.llm.pricing import cost_usd

    approx_input = (len(system) + len(user)) // 4
    return cost_usd(provider, model, input_tokens=approx_input, output_tokens=max_tokens)
