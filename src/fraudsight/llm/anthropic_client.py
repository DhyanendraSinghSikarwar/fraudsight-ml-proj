"""Anthropic provider.

Three choices worth defending:

* **`output_config.format` + client-side Pydantic validation**, rather than the
  `messages.parse()` helper. The helper hides the schema, and we need the raw
  JSON in hand to feed a *validation error* back into the repair loop. See
  llm/jsonschema.py for the two-layer constraint split.

* **`effort` is the latency lever, not `thinking: disabled`.** The triage path
  has a p95 SLA of 6s. Turning thinking off on Opus 5 is the more expensive
  choice in every sense -- it can leak internal tags into the response -- while
  `effort: "low"` gets most of the latency and token saving with thinking left
  on. The judge runs at `high`, where correctness beats latency.

* **Refusal failover is the router's job, not the server's.** Anthropic's
  server-side `fallbacks` parameter covers policy refusals only. Our triage path
  also has to survive timeouts, 5xx, rate limits, and a daily budget breach, and
  it must fail over on all of them with one consistent code path and one set of
  traces. A refusal here raises `ProviderRefusal`, which the router treats as a
  failover trigger like any other -- see llm/router.py.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from fraudsight.llm.base import (
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTimeout,
    ProviderUnavailable,
    SchemaValidationError,
    Usage,
)
from fraudsight.llm.jsonschema import describe_violation, schema_for
from fraudsight.llm.pricing import cost_usd

T = TypeVar("T", bound=BaseModel)

# Effort tiers per call site. Both keep adaptive thinking on.
EFFORT_TRIAGE = "low"
EFFORT_JUDGE = "high"


class AnthropicClient:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        timeout_s: float = 8.0,
        effort: str = EFFORT_TRIAGE,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ProviderError(
                "anthropic SDK not installed. `uv pip install -e '.[llm]'`",
                provider=self.provider_name,
            ) from exc

        self._client = AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            timeout=timeout_s,
            max_retries=0,  # retry policy belongs to the router, not two layers of it
        )
        self.model = model
        self.effort = effort
        self._timeout_s = timeout_s

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._system_blocks(system, cache_system),
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema_for(schema)},
                "effort": self.effort,
            },
        }
        text, usage = await self._call(payload)

        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as exc:
            # Surfaced with the field-level detail so the repair node can quote
            # the actual rule that was broken.
            raise SchemaValidationError(
                describe_violation(exc, schema.__name__), provider=self.provider_name
            ) from exc
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"output was not valid JSON: {exc}", provider=self.provider_name
            ) from exc
        return parsed, usage

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
    ) -> tuple[str, Usage]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._system_blocks(system, cache_system),
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        return await self._call(payload)

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        # Anthropic exposes no embeddings endpoint. Rather than silently produce
        # a degenerate vector, say so -- the retrieval tool then routes to the
        # configured embedding provider (see tools/similar_cases.py).
        raise ProviderError(
            "anthropic has no embeddings endpoint; configure an embedding provider",
            provider=self.provider_name,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _system_blocks(system: str, cache: bool) -> list[dict[str, Any]]:
        """System prompt as a cacheable block.

        The triage system prompt is static and well over the 512-token minimum
        for Opus 5, and triage runs thousands of times a day against a
        byte-identical prefix -- so the cache read is close to free money. The
        per-transaction evidence goes in the user turn, *after* the breakpoint,
        which is what keeps the prefix stable.
        """
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    async def _call(self, payload: dict[str, Any]) -> tuple[str, Usage]:
        import anthropic

        t0 = time.perf_counter()
        try:
            response = await self._client.messages.create(**payload)
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeout(str(exc), provider=self.provider_name) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimited(str(exc), provider=self.provider_name) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
            raise ProviderError(
                f"{exc.status_code}: {exc.message}", provider=self.provider_name
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = self._usage(response, latency_ms)

        # Check stop_reason BEFORE touching content: on a refusal, content is
        # empty (pre-output) or a partial to be discarded (mid-stream).
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            usage.refused = True
            raise ProviderRefusal(
                f"provider declined the request (category={category})",
                provider=self.provider_name,
                category=category,
            )

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise SchemaValidationError(
                f"empty response body (stop_reason={getattr(response, 'stop_reason', None)})",
                provider=self.provider_name,
            )
        return text, usage

    def _usage(self, response: Any, latency_ms: float) -> Usage:
        u = getattr(response, "usage", None)
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        model = getattr(response, "model", self.model)
        return Usage(
            provider=self.provider_name,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            cost_usd=cost_usd(
                self.provider_name,
                model,
                input_tokens=inp,
                output_tokens=out,
                cache_read_tokens=cr,
                cache_write_tokens=cw,
            ),
            latency_ms=round(latency_ms, 3),
        )
