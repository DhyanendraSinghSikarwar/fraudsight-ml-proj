"""Cascade, fallback, circuit breaker, and budget enforcement in one place.

Business logic calls `LLMRouter.structured(...)` and never learns which provider
served it. Everything that makes a production LLM call survivable lives here:

* **Retry only what is retryable.** A 429 wants backoff; a policy refusal will
  refuse identically on the second attempt and retrying it just spends money.
  The error taxonomy in `llm/base.py` carries `retryable` / `trips_breaker` so
  this module never inspects vendor exception types.

* **Circuit breaker per provider, not global.** When Anthropic is timing out,
  the breaker opens on *that* client and traffic moves to the fallback
  immediately -- without a breaker, every request pays the full timeout before
  failing over, and the p95 SLA is gone even though a healthy provider is
  sitting right there.

* **A budget breach does not fail over.** Failing over to a second paid provider
  when the cap is hit turns one bill into two. `BudgetExceeded` walks straight
  down to the deterministic provider, which is free.

* **Degradation is recorded, not hidden.** Every `Usage` that came from anything
  other than the primary carries `degraded=True`, so "how often did we serve a
  brief from the fallback?" is a query rather than an archaeology exercise. A
  system that silently degrades is a system whose quality numbers are fiction.

The ladder is ordered cheapest-recovery-first: retry the primary, then each
fallback in turn, then the deterministic provider. It only ever runs out of
options if the deterministic provider itself raises, which would be a code bug.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from fraudsight.llm.base import (
    BudgetExceeded,
    LLMClient,
    ProviderError,
    ProviderTimeout,
    Usage,
)
from fraudsight.llm.usage import CostLedger, estimate_cost

T = TypeVar("T", bound=BaseModel)


@dataclass
class Breaker:
    """One provider's health. Opens after `threshold` consecutive trip-worthy
    failures, closes after `cooldown_s` of quiet."""

    threshold: int = 3
    cooldown_s: float = 30.0
    failures: int = 0
    open_until: float = 0.0
    opened_count: int = 0

    def is_open(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) < self.open_until

    def record_failure(self, now: float | None = None) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = (now or time.monotonic()) + self.cooldown_s
            self.failures = 0
            self.opened_count += 1

    def record_success(self) -> None:
        # Reset on success rather than decay: an intermittent single failure
        # inside otherwise healthy traffic is not a reason to trip.
        self.failures = 0


@dataclass
class Attempt:
    """One provider attempt, kept for the trace."""

    provider: str
    model: str
    outcome: str  # ok | retry | failover | skipped_breaker | skipped_budget
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class RouteResult:
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return len(self.attempts) > 1

    def as_dict(self) -> dict[str, object]:
        return {
            "degraded": self.degraded,
            "attempts": [a.__dict__ for a in self.attempts],
        }


class LLMRouter:
    def __init__(
        self,
        primary: LLMClient,
        fallbacks: list[LLMClient] | None = None,
        *,
        ledger: CostLedger | None = None,
        timeout_s: float = 8.0,
        max_attempts: int = 3,
        breaker_cooldown_s: float = 30.0,
    ) -> None:
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.ledger = ledger
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self._breakers: dict[str, Breaker] = {
            c.provider_name: Breaker(cooldown_s=breaker_cooldown_s)
            for c in [primary, *self.fallbacks]
        }
        self.last_route = RouteResult()

    # ------------------------------------------------------------------

    @property
    def chain(self) -> list[LLMClient]:
        return [self.primary, *self.fallbacks]

    def breaker(self, provider: str) -> Breaker:
        return self._breakers.setdefault(provider, Breaker())

    def health(self) -> dict[str, dict[str, object]]:
        now = time.monotonic()
        return {
            name: {
                "open": b.is_open(now),
                "consecutive_failures": b.failures,
                "times_opened": b.opened_count,
            }
            for name, b in self._breakers.items()
        }

    # ------------------------------------------------------------------

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4096,
        operation: str = "structured",
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        async def call(client: LLMClient) -> tuple[T, Usage]:
            return await client.structured(
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
                cache_system=cache_system,
            )

        return await self._run(call, system=system, user=user, max_tokens=max_tokens,
                               operation=operation)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        operation: str = "complete",
        cache_system: bool = True,
    ) -> tuple[str, Usage]:
        async def call(client: LLMClient) -> tuple[str, Usage]:
            return await client.complete(
                system=system, user=user, max_tokens=max_tokens, cache_system=cache_system
            )

        return await self._run(call, system=system, user=user, max_tokens=max_tokens,
                               operation=operation)

    async def embed(
        self, texts: list[str], *, operation: str = "embed"
    ) -> tuple[list[list[float]], Usage]:
        async def call(client: LLMClient) -> tuple[list[list[float]], Usage]:
            return await client.embed(texts)

        return await self._run(
            call, system="", user="\n".join(texts), max_tokens=0, operation=operation
        )

    # ------------------------------------------------------------------

    async def _run(self, call, *, system: str, user: str, max_tokens: int, operation: str):  # type: ignore[no-untyped-def]
        route = RouteResult()
        self.last_route = route
        last_error: Exception | None = None
        budget_hit = False

        for position, client in enumerate(self.chain):
            name = client.provider_name
            breaker = self.breaker(name)

            if breaker.is_open():
                route.attempts.append(Attempt(name, _model_of(client), "skipped_breaker"))
                continue

            # Once the cap is hit, only a free provider may serve the call.
            if budget_hit and not _is_free(client):
                route.attempts.append(Attempt(name, _model_of(client), "skipped_budget"))
                continue

            attempts_here = self.max_attempts if position == 0 else 1
            for attempt_no in range(attempts_here):
                reserved = 0.0
                t0 = time.perf_counter()
                try:
                    if self.ledger is not None and not _is_free(client):
                        reserved = self.ledger.reserve(
                            estimate_cost(
                                name,
                                _model_of(client),
                                system=system,
                                user=user,
                                max_tokens=max_tokens,
                            )
                        )
                    result, usage = await asyncio.wait_for(call(client), timeout=self.timeout_s)
                except TimeoutError as exc:
                    if self.ledger is not None:
                        self.ledger.release(reserved)
                    last_error = ProviderTimeout(
                        f"router timeout after {self.timeout_s}s", provider=name
                    )
                    breaker.record_failure()
                    route.attempts.append(
                        Attempt(name, _model_of(client), "failover", str(last_error), _ms(t0))
                    )
                    break  # a timing-out provider will not answer faster on retry
                except BudgetExceeded as exc:
                    if self.ledger is not None:
                        self.ledger.release(reserved)
                    budget_hit = True
                    last_error = exc
                    route.attempts.append(
                        Attempt(name, _model_of(client), "skipped_budget", str(exc), _ms(t0))
                    )
                    break
                except ProviderError as exc:
                    if self.ledger is not None:
                        self.ledger.release(reserved)
                    last_error = exc
                    if exc.trips_breaker:
                        breaker.record_failure()
                    is_last_local = attempt_no == attempts_here - 1
                    outcome = "failover" if (is_last_local or not exc.retryable) else "retry"
                    route.attempts.append(
                        Attempt(name, _model_of(client), outcome, str(exc), _ms(t0))
                    )
                    if not exc.retryable:
                        break
                    if not is_last_local:
                        # Exponential backoff with a deterministic schedule --
                        # jitter would make the eval suite non-reproducible, and
                        # a single worker has no thundering-herd problem to
                        # solve. Documented in docs/adr/0008.
                        await asyncio.sleep(0.25 * (2**attempt_no))
                    continue
                else:
                    breaker.record_success()
                    usage.degraded = position > 0 or len(route.attempts) > 0
                    route.attempts.append(
                        Attempt(name, usage.model, "ok", None, usage.latency_ms)
                    )
                    if self.ledger is not None:
                        self.ledger.record(usage, operation=operation, reserved=reserved)
                    return result, usage

        raise last_error or ProviderError("no provider available", provider="router")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def _model_of(client: LLMClient) -> str:
    return str(getattr(client, "model", "unknown"))


def _is_free(client: LLMClient) -> bool:
    return client.provider_name == "deterministic"
