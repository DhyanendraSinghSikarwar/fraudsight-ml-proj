"""Token pricing table and cost arithmetic.

Prices move. This table is the single place they live, it records the date it
was checked, and `cost_per_triage` in the daily report is computed from measured
token counts rather than an assumed average -- so a price change or a prompt
that quietly doubled in size both show up in the cost dashboard instead of
being absorbed silently.

Rates are USD per million tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_AS_OF = "2026-06-24"


@dataclass(frozen=True)
class Rate:
    input_per_mtok: float
    output_per_mtok: float
    # Cache reads are ~0.1x input; cache writes ~1.25x input (5-minute TTL).
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


# Anthropic first-party rates.
ANTHROPIC: dict[str, Rate] = {
    "claude-opus-5": Rate(5.00, 25.00),
    "claude-opus-4-8": Rate(5.00, 25.00),
    "claude-sonnet-5": Rate(3.00, 15.00),
    "claude-sonnet-4-6": Rate(3.00, 15.00),
    "claude-haiku-4-5": Rate(1.00, 5.00),
    "claude-fable-5": Rate(10.00, 50.00),
}

OPENAI: dict[str, Rate] = {
    "gpt-4.1": Rate(2.00, 8.00),
    "gpt-4.1-mini": Rate(0.40, 1.60),
    "gpt-4.1-nano": Rate(0.10, 0.40),
    "text-embedding-3-small": Rate(0.02, 0.0),
    "text-embedding-3-large": Rate(0.13, 0.0),
}

# The deterministic provider runs locally. Zero is the honest number, and having
# it in the table keeps the cost code free of provider special-casing.
LOCAL: dict[str, Rate] = {"deterministic-v1": Rate(0.0, 0.0), "hash-embed-v1": Rate(0.0, 0.0)}

_TABLES = {"anthropic": ANTHROPIC, "openai": OPENAI, "deterministic": LOCAL}

# Fallback for an unrecognised model: Opus-tier rates. Erring expensive means an
# unpriced model shows up as a cost spike rather than as free capacity.
UNKNOWN_MODEL_RATE = Rate(5.00, 25.00)


def rate_for(provider: str, model: str) -> Rate:
    return _TABLES.get(provider, {}).get(model, UNKNOWN_MODEL_RATE)


def is_priced(provider: str, model: str) -> bool:
    return model in _TABLES.get(provider, {})


def cost_usd(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    r = rate_for(provider, model)
    m = 1_000_000.0
    total = (
        input_tokens * r.input_per_mtok
        + output_tokens * r.output_per_mtok
        + cache_read_tokens * r.input_per_mtok * r.cache_read_multiplier
        + cache_write_tokens * r.input_per_mtok * r.cache_write_multiplier
    ) / m
    return round(total, 8)
