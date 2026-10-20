"""The router's failure behaviour, which is the part that matters in an incident."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from fraudsight.llm.base import (
    BudgetExceeded,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTimeout,
    ProviderUnavailable,
    Usage,
)
from fraudsight.llm.router import LLMRouter
from fraudsight.llm.usage import CostLedger


class Answer(BaseModel):
    value: str


class FakeClient:
    """A scriptable provider. `script` is consumed one entry per call: an
    exception class to raise, or a string to return."""

    def __init__(self, name: str, script: list[object], model: str = "fake-1") -> None:
        self.provider_name = name
        self.model = model
        self.script = list(script)
        self.calls = 0

    async def structured(self, *, system, user, schema, max_tokens=4096, cache_system=True):  # type: ignore[no-untyped-def]
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted failure", provider=self.provider_name)
        if step == "hang":
            await asyncio.sleep(10)
        return schema(value=str(step)), Usage(
            provider=self.provider_name, model=self.model, input_tokens=10,
            output_tokens=5, cost_usd=0.001,
        )

    async def complete(self, **kw):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def embed(self, texts: Sequence[str]):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def _call(router: LLMRouter):  # type: ignore[no-untyped-def]
    return router.structured(system="s", user="u", schema=Answer)


async def test_retries_retryable_errors_on_the_primary() -> None:
    primary = FakeClient("primary", [ProviderRateLimited, ProviderRateLimited, "recovered"])
    router = LLMRouter(primary, [], ledger=None, max_attempts=3)
    answer, usage = await _call(router)
    assert answer.value == "recovered"
    assert primary.calls == 3
    assert usage.degraded is True  # it took more than one attempt; say so


async def test_does_not_retry_a_refusal_and_fails_over() -> None:
    """A policy refusal will refuse identically the second time. Retrying it
    spends money to learn nothing."""
    primary = FakeClient("primary", [ProviderRefusal])
    fallback = FakeClient("fallback", ["from_fallback"])
    router = LLMRouter(primary, [fallback], ledger=None, max_attempts=3)
    answer, usage = await _call(router)
    assert answer.value == "from_fallback"
    assert primary.calls == 1
    assert usage.degraded is True


async def test_breaker_opens_and_diverts_without_paying_the_timeout() -> None:
    primary = FakeClient("primary", [ProviderUnavailable] * 6)
    fallback = FakeClient("fallback", ["a", "b", "c"])
    router = LLMRouter(primary, [fallback], ledger=None, max_attempts=3, breaker_cooldown_s=60)

    await _call(router)
    assert router.breaker("primary").is_open(), "3 trip-worthy failures must open the breaker"

    calls_before = primary.calls
    await _call(router)
    assert primary.calls == calls_before, "an open breaker must skip the provider entirely"
    assert router.last_route.attempts[0].outcome == "skipped_breaker"


async def test_breaker_resets_on_success() -> None:
    primary = FakeClient("primary", [ProviderUnavailable, "ok"])
    router = LLMRouter(primary, [], ledger=None, max_attempts=3)
    await _call(router)
    assert router.breaker("primary").failures == 0


async def test_router_timeout_fails_over_rather_than_hanging() -> None:
    primary = FakeClient("primary", ["hang"])
    fallback = FakeClient("fallback", ["quick"])
    router = LLMRouter(primary, [fallback], ledger=None, timeout_s=0.05, max_attempts=2)
    answer, _ = await _call(router)
    assert answer.value == "quick"


async def test_budget_breach_does_not_fail_over_to_another_paid_provider() -> None:
    """Failing over on a denial-of-wallet turns one bill into two. Only a free
    provider may serve a call after the cap is hit."""
    ledger = CostLedger(daily_budget_usd=0.0000001)
    paid_primary = FakeClient("primary", ["never_reached"])
    paid_fallback = FakeClient("openai", ["also_never_reached"])
    free = FakeClient("deterministic", ["free_result"])
    router = LLMRouter(paid_primary, [paid_fallback, free], ledger=ledger, max_attempts=2)

    answer, _ = await _call(router)
    assert answer.value == "free_result"
    assert paid_primary.calls == 0
    assert paid_fallback.calls == 0
    outcomes = [a.outcome for a in router.last_route.attempts]
    assert outcomes.count("skipped_budget") == 2


async def test_exhausting_every_provider_raises_the_last_error() -> None:
    primary = FakeClient("primary", [ProviderTimeout] * 4)
    fallback = FakeClient("fallback", [ProviderUnavailable] * 2)
    router = LLMRouter(primary, [fallback], ledger=None, max_attempts=2)
    with pytest.raises((ProviderTimeout, ProviderUnavailable)):
        await _call(router)


async def test_ledger_records_and_enforces() -> None:
    # The budget has to clear the *pessimistic* pre-call reservation, which
    # assumes the response fills max_tokens. Sizing it to the actual cost would
    # make this test fail on the reservation rather than on the thing it checks.
    ledger = CostLedger(daily_budget_usd=1.0)
    client = FakeClient("primary", ["a", "b", "c", "d"])
    router = LLMRouter(client, [], ledger=ledger, max_attempts=1)

    await _call(router)
    assert ledger.totals().calls == 1
    assert ledger.spent_today() == pytest.approx(0.001)
    # The reservation is released once the real usage lands, so a stream of
    # cheap calls does not drain the budget at the estimated rate.
    assert ledger.remaining_today() == pytest.approx(1.0 - 0.001)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(estimate_usd=10.0)


def test_ledger_releases_in_flight_reservations() -> None:
    """A reservation that is never released permanently shrinks the budget --
    a slow leak that looks like a budget breach hours later."""
    ledger = CostLedger(daily_budget_usd=1.0)
    held = ledger.reserve(0.4)
    assert ledger.remaining_today() == pytest.approx(0.6)
    ledger.release(held)
    assert ledger.remaining_today() == pytest.approx(1.0)
