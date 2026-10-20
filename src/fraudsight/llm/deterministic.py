"""A rule-based provider that satisfies the `LLMClient` protocol.

This is not a mock and it is not a stub. It is a shipped, scored component with
three real jobs:

1. **The last rung of the fallback ladder.** When the primary provider is down,
   the fallback is down, and the daily budget is spent, the triage path still
   has to emit a valid `TriageBrief` -- because the alternative is dropping a
   transaction, which is the one thing the NFRs forbid. Its briefs are worse
   than an LLM's; they are never invalid.

2. **A measurable floor for the evals.** Every metric in the eval report is also
   computed against this provider. "The agent scores 0.89 agreement" means
   nothing on its own; "the agent scores 0.89 where the rule-based floor scores
   0.71" is a claim about what the LLM is contributing. Portfolio agents almost
   never publish the floor, which is why their numbers are unreadable.

3. **A zero-credential CI path.** The eval gate runs on every pull request. A
   gate that needs a vendor key is a gate that gets disabled.

The honest limitation, stated rather than buried: this provider cannot *reason*.
It ranks pre-extracted signals by weight and renders them into a template. It
scores well on schema compliance and groundedness (it can only cite what it was
handed) and poorly on narrative quality and on the ambiguous slice -- which is
exactly the shape you would predict, and exactly what the eval report shows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from fraudsight.llm.base import SchemaValidationError, Usage
from fraudsight.llm.jsonschema import describe_violation
from fraudsight.llm.pricing import cost_usd

T = TypeVar("T", bound=BaseModel)

MODEL_NAME = "deterministic-v1"
EMBED_MODEL_NAME = "hash-embed-v1"

# The prompt builders wrap a machine-readable copy of the evidence bundle in
# this tag. An LLM ignores it (the same facts are in the prose above it); this
# provider parses it. Keeping the contract explicit is what stops the rule-based
# path from drifting into "parse the English prompt with regexes".
EVIDENCE_TAG = "evidence_json"
_EVIDENCE_RE = re.compile(
    rf"<{EVIDENCE_TAG}>\s*(?P<body>\{{.*?\}})\s*</{EVIDENCE_TAG}>", re.DOTALL
)

Synthesiser = Callable[[dict[str, Any], str, str], dict[str, Any]]

_SYNTHESISERS: dict[str, Synthesiser] = {}


def register(schema_name: str) -> Callable[[Synthesiser], Synthesiser]:
    """Bind a schema to the rule that produces it.

    A registry rather than a generic "fill in the schema with defaults" walker:
    a walker would emit structurally valid nonsense and the eval suite would
    score it as if it meant something.
    """

    def deco(fn: Synthesiser) -> Synthesiser:
        _SYNTHESISERS[schema_name] = fn
        return fn

    return deco


def extract_evidence(user: str) -> dict[str, Any]:
    m = _EVIDENCE_RE.search(user)
    if not m:
        return {}
    try:
        return json.loads(m.group("body"))
    except json.JSONDecodeError:
        return {}


class DeterministicClient:
    """Rule-based `LLMClient`. No network, no key, no variance across runs."""

    provider_name = "deterministic"

    def __init__(self, model: str = MODEL_NAME, *, latency_ms: float = 0.0) -> None:
        self.model = model
        # Optional synthetic latency so the replay harness can exercise timeout
        # and concurrency behaviour without a live provider.
        self._latency_ms = latency_ms

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        t0 = time.perf_counter()
        synth = _SYNTHESISERS.get(schema.__name__)
        if synth is None:
            raise SchemaValidationError(
                f"no deterministic rule registered for {schema.__name__}. Register one "
                f"with @deterministic.register('{schema.__name__}') or run against a "
                f"real provider.",
                provider=self.provider_name,
            )
        payload = synth(extract_evidence(user), system, user)
        try:
            parsed = schema.model_validate(payload)
        except ValidationError as exc:
            # A bug in a rule is a bug in shipped code, so it surfaces as the
            # same error type the repair loop already handles rather than
            # crashing the triage run.
            raise SchemaValidationError(
                describe_violation(exc, schema.__name__), provider=self.provider_name
            ) from exc
        return parsed, self._usage(system, user, json.dumps(payload), t0)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
    ) -> tuple[str, Usage]:
        t0 = time.perf_counter()
        ev = extract_evidence(user)
        text = json.dumps(ev, sort_keys=True) if ev else user[-max_tokens * 4 :]
        return text, self._usage(system, user, text, t0)

    async def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], Usage]:
        """Hashed bag-of-tokens embedding.

        Not a semantic embedding and not presented as one -- it captures lexical
        overlap only. That is enough for the similar-case tool to return
        genuinely related fraud narratives (they share signal vocabulary:
        `velocity_spike`, `device_unseen`, `billing_shipping_mismatch`), and the
        retrieval precision the eval suite reports is measured, not assumed.
        Swapping in a real embedding provider is a config change; the retrieval
        numbers for both are in the eval report.
        """
        t0 = time.perf_counter()
        dim = _embed_dim()
        vectors = [_hash_embed(t, dim) for t in texts]
        joined = "\n".join(texts)
        return vectors, self._usage("", joined, "", t0, model=EMBED_MODEL_NAME)

    async def aclose(self) -> None:
        return None

    def _usage(
        self, system: str, user: str, output: str, t0: float, *, model: str | None = None
    ) -> Usage:
        # Token counts are a chars/4 approximation. They exist so the cost
        # dashboard has the same shape on every provider; the cost is genuinely
        # zero because the work happened in this process.
        name = model or self.model
        inp = (len(system) + len(user)) // 4
        out = len(output) // 4
        return Usage(
            provider=self.provider_name,
            model=name,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost_usd(self.provider_name, name, input_tokens=inp, output_tokens=out),
            latency_ms=round((time.perf_counter() - t0) * 1000.0 + self._latency_ms, 3),
        )


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def _embed_dim() -> int:
    from fraudsight.config import settings

    return settings.embedding_dim


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _hash_embed(text: str, dim: int) -> list[float]:
    """Signed hashing trick with sublinear term weighting, L2-normalised.

    The sign bit matters: without it, unrelated tokens colliding on the same
    bucket can only ever increase similarity, and every pair of long documents
    looks related.
    """
    vec = [0.0] * dim
    counts: dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        counts[tok] = counts.get(tok, 0) + 1
    for tok, n in counts.items():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(n))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Both sides are already unit-norm, so this is a dot product; the guard is
    for hand-built vectors in tests."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb) if abs(na * nb - 1.0) > 1e-6 else dot
