"""Provider wiring. The one place that knows which vendor SDK exists.

`build_router()` reads config and assembles the fallback ladder. Every caller in
the system takes an `LLMRouter` and is therefore provider-agnostic by
construction -- proving the "model swap is a config change" claim is a matter of
setting `PRIMARY_PROVIDER` and re-running the evals, which is what
tests/integration/test_provider_swap.py does.
"""

from __future__ import annotations

from fraudsight.config import Provider, Settings, settings as default_settings
from fraudsight.llm.base import (
    BudgetExceeded,
    LLMClient,
    Message,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTimeout,
    ProviderUnavailable,
    SchemaValidationError,
    Usage,
)
from fraudsight.llm.deterministic import DeterministicClient
from fraudsight.llm.router import LLMRouter
from fraudsight.llm.usage import CostLedger

__all__ = [
    "BudgetExceeded",
    "CostLedger",
    "DeterministicClient",
    "LLMClient",
    "LLMRouter",
    "Message",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderRefusal",
    "ProviderTimeout",
    "ProviderUnavailable",
    "SchemaValidationError",
    "Usage",
    "build_client",
    "build_ledger",
    "build_router",
]


def build_client(
    provider: Provider,
    model: str,
    *,
    cfg: Settings | None = None,
    effort: str | None = None,
) -> LLMClient:
    cfg = cfg or default_settings
    if provider == "deterministic":
        return DeterministicClient()

    key = cfg.key_for(provider)
    if key is None:
        raise ProviderError(f"no API key configured for {provider}", provider=provider)

    if provider == "anthropic":
        from fraudsight.llm.anthropic_client import EFFORT_TRIAGE, AnthropicClient

        return AnthropicClient(
            key, model, timeout_s=cfg.triage_timeout_s, effort=effort or EFFORT_TRIAGE
        )
    if provider == "openai":
        from fraudsight.llm.openai_client import OpenAIClient

        return OpenAIClient(key, model, timeout_s=cfg.triage_timeout_s)
    raise ProviderError(f"unknown provider {provider!r}", provider=str(provider))


def build_ledger(cfg: Settings | None = None) -> CostLedger:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    return CostLedger(
        daily_budget_usd=cfg.daily_llm_budget_usd,
        alert_fraction=cfg.budget_alert_fraction,
        ledger_path=cfg.artifact_dir / "cost_ledger.jsonl",
    )


def build_router(
    cfg: Settings | None = None,
    *,
    ledger: CostLedger | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> LLMRouter:
    """Primary -> configured fallback -> deterministic.

    The deterministic provider is appended unconditionally and cannot be
    configured away. That is deliberate: `graceful degradation` in the NFR table
    is only true if the last rung always exists.
    """
    cfg = cfg or default_settings
    primary_model = model or cfg.triage_model

    try:
        primary = build_client(cfg.primary_provider, primary_model, cfg=cfg, effort=effort)
    except ProviderError:
        # Configured for a provider we hold no key for. Rather than crash at
        # import time, degrade to deterministic and let the run report itself as
        # degraded -- a missing key on a laptop should not stop `make demo`.
        primary = DeterministicClient()

    fallbacks: list[LLMClient] = []
    if cfg.fallback_provider and cfg.fallback_provider != cfg.primary_provider:
        try:
            fallbacks.append(
                build_client(cfg.fallback_provider, cfg.fallback_model, cfg=cfg, effort=effort)
            )
        except ProviderError:
            pass

    if not any(c.provider_name == "deterministic" for c in [primary, *fallbacks]):
        fallbacks.append(DeterministicClient())

    return LLMRouter(
        primary,
        fallbacks,
        ledger=ledger if ledger is not None else build_ledger(cfg),
        timeout_s=cfg.triage_timeout_s,
        breaker_cooldown_s=cfg.circuit_breaker_cooldown_s,
    )
