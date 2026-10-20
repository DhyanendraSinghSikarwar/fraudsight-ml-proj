"""OpenAI provider.

Exists to make the abstraction real. A "provider-agnostic" design with one
implementation is a claim, not an architecture; the fallback path in
llm/router.py is exercised in tests against this client, and it is also the
embeddings provider for similar-case retrieval (Anthropic has no embeddings
endpoint).
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
    ProviderTimeout,
    ProviderUnavailable,
    SchemaValidationError,
    Usage,
)
from fraudsight.llm.jsonschema import describe_violation, schema_for
from fraudsight.llm.pricing import cost_usd

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    provider_name = "openai"

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        timeout_s: float = 8.0,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 256,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ProviderError(
                "openai SDK not installed. `uv pip install -e '.[llm]'`",
                provider=self.provider_name,
            ) from exc

        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value(), timeout=timeout_s, max_retries=0
        )
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema_for(schema),
            },
        }
        text, usage = await self._chat(system, user, max_tokens, response_format)
        try:
            return schema.model_validate_json(text), usage
        except ValidationError as exc:
            raise SchemaValidationError(
                describe_violation(exc, schema.__name__), provider=self.provider_name
            ) from exc
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"output was not valid JSON: {exc}", provider=self.provider_name
            ) from exc

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
    ) -> tuple[str, Usage]:
        return await self._chat(system, user, max_tokens, None)

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        import openai

        t0 = time.perf_counter()
        try:
            resp = await self._client.embeddings.create(
                model=self.embedding_model,
                input=list(texts),
                dimensions=self.embedding_dim,
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeout(str(exc), provider=self.provider_name) from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimited(str(exc), provider=self.provider_name) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
            raise ProviderError(str(exc), provider=self.provider_name) from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
        usage = Usage(
            provider=self.provider_name,
            model=self.embedding_model,
            input_tokens=tokens,
            cost_usd=cost_usd(
                self.provider_name, self.embedding_model, input_tokens=tokens, output_tokens=0
            ),
            latency_ms=round(latency_ms, 3),
        )
        return [d.embedding for d in resp.data], usage

    async def aclose(self) -> None:
        await self._client.close()

    async def _chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise ProviderTimeout(str(exc), provider=self.provider_name) from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimited(str(exc), provider=self.provider_name) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
            raise ProviderError(str(exc), provider=self.provider_name) from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise ProviderError("content filtered by provider", provider=self.provider_name)

        text = choice.message.content or ""
        if not text.strip():
            raise SchemaValidationError(
                f"empty response body (finish_reason={choice.finish_reason})",
                provider=self.provider_name,
            )

        u = resp.usage
        inp = int(getattr(u, "prompt_tokens", 0) or 0)
        out = int(getattr(u, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)

        usage = Usage(
            provider=self.provider_name,
            model=resp.model,
            input_tokens=max(inp - cached, 0),
            output_tokens=out,
            cache_read_tokens=cached,
            cost_usd=cost_usd(
                self.provider_name,
                resp.model,
                input_tokens=max(inp - cached, 0),
                output_tokens=out,
                cache_read_tokens=cached,
            ),
            latency_ms=round(latency_ms, 3),
        )
        return text, usage
