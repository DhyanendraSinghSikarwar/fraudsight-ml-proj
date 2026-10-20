# ADR 0008 — Retry only what is retryable; never fail over on a budget breach

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** how the system behaves when a provider misbehaves

## Context

Four things can go wrong with an LLM call and they want four different
responses: rate limits, timeouts and 5xx, policy refusals, and a spent budget.
The common design treats them as one "error" and retries all of them, which is
wrong in three of the four cases.

## Decision

The error taxonomy lives in `llm/base.py` and carries the policy as data:

| Error | Retryable | Trips breaker | Fails over |
|---|---|---|---|
| `ProviderRateLimited` | yes | **no** | after local retries |
| `ProviderTimeout` | yes | yes | immediately |
| `ProviderUnavailable` (5xx, conn) | yes | yes | after local retries |
| `ProviderRefusal` (200, declined) | **no** | no | yes |
| `SchemaValidationError` | yes | no | after local retries |
| `BudgetExceeded` | **no** | no | **only to a free provider** |

The router reads these flags. It never inspects a vendor exception type, which
is what keeps adding a third provider from being a change to the routing logic.

## Consequences

**Rate limits back off but do not trip the breaker.** A 429 means "slow down",
not "this provider is broken". Tripping on it would move traffic to a fallback
that is about to receive the same burst.

**Timeouts break out of the local retry loop immediately.** A provider that took
8 seconds to not answer will not answer faster on the second try, and the p95
budget is 6 seconds total.

**Refusals are never retried against the same model.** The classifier that
declined will decline again. They *are* worth failing over, because a different
model carries different classifiers.

**A budget breach does not fail over to a second paid provider.** This is the
one that matters most and the one most systems get wrong: failing over on a
denial-of-wallet turns one bill into two. `BudgetExceeded` walks straight down
to the deterministic provider, which is free.
`tests/unit/test_router.py::test_budget_breach_does_not_fail_over_to_another_paid_provider`
asserts it.

**Per-provider circuit breakers, not a global one.** With a global breaker,
every request pays the full timeout before failing over, and the SLA is gone
while a healthy provider sits idle. Three consecutive trip-worthy failures open
a provider for 30 seconds; the next request skips it entirely and the trace
records `skipped_breaker`.

**Degradation is recorded, never hidden.** Any `Usage` served by anything other
than the primary carries `degraded=True`, so "how often did we serve a brief
from the fallback?" is a query. The daily report alerts above 5%.

**Backoff is deterministic** — 0.25s, 0.5s, 1.0s, no jitter. Jitter would make
the eval suite non-reproducible, and a single worker has no thundering herd to
solve. Add jitter when this runs on more than a handful of workers.

## Alternatives rejected

**Tenacity decorators on the client methods.** Retry has to be aware of the
circuit breaker, the budget ledger and the failover chain; splitting it across
a decorator and the router puts the policy in two places.

**The provider SDK's built-in retries.** Both SDKs are configured with
`max_retries=0` deliberately. Two retry layers multiply: three SDK retries
inside three router attempts is nine calls and a 24-second worst case against a
6-second budget.

**Retry everything N times.** Spends money on refusals, wastes the latency
budget on timeouts, and hammers a rate-limited provider.

## When this flips

Multi-worker deployment needs jitter and a shared breaker state (Redis). Both
are small changes; neither is justified at one worker.
