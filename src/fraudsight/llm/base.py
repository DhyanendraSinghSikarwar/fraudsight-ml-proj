"""Provider-agnostic LLM contract.

Business logic never imports a vendor SDK. The triage graph depends on
`LLMClient`; `router.LLMRouter` decides which concrete client serves a call.
That is what makes "swap the model" a config change instead of a refactor, and
it is also what lets the eval suite run against a deterministic provider in CI
with no vendor account.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class Usage(BaseModel):
    """Per-call accounting. Every LLM call in the system produces one of these."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    # True when the call was served by a fallback rather than the primary.
    degraded: bool = False
    # Set when the provider declined the request on policy grounds. Distinct
    # from an error: an HTTP 200 with no usable content.
    refused: bool = False

    def merge(self, other: Usage) -> Usage:
        """Combine attempts within one logical operation (retries, repairs)."""
        return Usage(
            provider=other.provider,
            model=other.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
            latency_ms=round(self.latency_ms + other.latency_ms, 3),
            degraded=self.degraded or other.degraded,
            refused=self.refused or other.refused,
        )


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """The whole surface the application is allowed to use.

    Deliberately narrow. A wide protocol leaks vendor concepts (tool blocks,
    thinking configs, beta headers) into business logic and makes the second
    implementation expensive -- which is how "provider-agnostic" designs end up
    with exactly one provider.
    """

    provider_name: str

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        """Return a schema-validated object. Raises on unrecoverable failure."""
        ...

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
    ) -> tuple[str, Usage]:
        ...

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        ...

    async def aclose(self) -> None:
        ...


# --------------------------------------------------------------------------
# Errors. The distinction matters for routing: only some are worth a retry,
# and only some should trip the circuit breaker.
# --------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base for provider failures."""

    retryable = False
    trips_breaker = False

    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message)
        self.provider = provider


class ProviderTimeout(ProviderError):
    retryable = True
    trips_breaker = True


class ProviderRateLimited(ProviderError):
    retryable = True
    trips_breaker = False  # backing off is right; failing over is not


class ProviderUnavailable(ProviderError):
    """5xx / connection failure."""

    retryable = True
    trips_breaker = True


class ProviderRefusal(ProviderError):
    """Provider returned 200 but declined on policy grounds.

    Not retryable against the same model -- the classifier will decline again.
    Worth failing over, because a different model carries different classifiers.
    """

    retryable = False
    trips_breaker = False

    def __init__(self, message: str, *, provider: str = "unknown", category: str | None = None):
        super().__init__(message, provider=provider)
        self.category = category


class SchemaValidationError(ProviderError):
    """Model returned content that does not satisfy the requested schema."""

    retryable = True
    trips_breaker = False


class BudgetExceeded(ProviderError):
    """Daily spend cap hit. Deliberately not retryable and does not fail over --
    failing over to a second paid provider on a budget breach is how a
    denial-of-wallet turns into two bills."""

    retryable = False
    trips_breaker = False
