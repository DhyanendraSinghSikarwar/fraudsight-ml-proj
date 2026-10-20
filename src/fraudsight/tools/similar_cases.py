"""Similar-case retrieval tool.

The query is built from the *derived signal names* plus a short amount/product
descriptor, not from the raw transaction. That matters: embedding a raw
transaction row retrieves cases with similar dollar amounts, which is nearly
useless, whereas embedding `velocity_spike device_unseen thin_customer_history`
retrieves cases with a similar shape. The measured difference is in
artifacts/retrieval_metrics.json.

Retrieved cases are drawn from the training window only. A case for a
test-window transaction that could retrieve its own neighbours from the test
window would leak, and would leak in a way that flatters every Track B metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from fraudsight.tools.base import ToolResult, ToolSpec

DESCRIPTION = (
    "Retrieves resolved historical fraud cases whose signal pattern resembles this "
    "transaction's, using hybrid dense + lexical search over the case corpus. Returns "
    "each case's narrative, its confirmed outcome, and the signals it exhibited. Use "
    "these as base rates for a pattern, not as proof about this transaction: a similar "
    "case that turned out legitimate is as informative as one that turned out fraudulent."
)


class SearchBackend(Protocol):
    def search(
        self,
        query_vector: Any,
        query_text: str,
        *,
        top_k: int = 5,
        candidates: int = 50,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[Any]: ...


def make(
    store: SearchBackend,
    embed_fn: Any | None = None,
    *,
    top_k: int = 5,
) -> ToolSpec:
    """`embed_fn` is an async callable returning one vector per text.

    Optional: without it the tool runs lexical-only, which is the honest
    degradation when no embedding provider is configured. The result records
    which mode ran so the eval report can separate the two.
    """

    async def run(scope: Mapping[str, Any]) -> ToolResult:
        query_text = str(scope.get("query_text") or "").strip()
        if not query_text:
            return ToolResult(
                tool="similar_cases",
                found=False,
                fields={"reason": "no signal vocabulary derived for this transaction"},
                source="case corpus",
            )

        vector = None
        mode = "lexical_only"
        if embed_fn is not None:
            try:
                vectors, _usage = await embed_fn([query_text])
                vector = np.asarray(vectors[0], dtype=np.float32)
                mode = "hybrid"
            except Exception as exc:  # noqa: BLE001 - degrade, never fail the triage
                # Losing the dense half costs retrieval precision; losing the
                # tool entirely costs the agent its base rates.
                mode = f"lexical_only (embedding failed: {type(exc).__name__})"

        exclude = frozenset(str(scope.get("exclude_case_id") or ""))
        hits = store.search(
            vector,
            query_text,
            top_k=top_k,
            exclude_ids=exclude if scope.get("exclude_case_id") else frozenset(),
        )
        cases = [h.as_dict() for h in hits]
        fraud_like = sum(1 for c in cases if c.get("outcome") == "confirmed_fraud")
        return ToolResult(
            tool="similar_cases",
            found=bool(cases),
            fields={
                "query": query_text,
                "retrieval_mode": mode,
                "returned": len(cases),
                "confirmed_fraud_among_returned": fraud_like,
                "cases": cases,
            },
            source="case corpus (training window)",
        )

    return ToolSpec(
        name="similar_cases",
        description=DESCRIPTION,
        fn=run,
        scope_keys=("query_text",),
    )


def build_query(signal_names: list[str], transaction: Mapping[str, Any]) -> str:
    """Signal vocabulary first, coarse descriptors second.

    Amount is bucketed rather than exact -- an exact amount is a near-unique
    token that pulls the lexical retriever toward coincidence.
    """
    amount = float(transaction.get("amount") or 0.0)
    bucket = (
        "micro" if amount < 5 else
        "small" if amount < 50 else
        "medium" if amount < 250 else
        "large" if amount < 1000 else
        "very_large"
    )
    product = str(transaction.get("product_code") or "unknown")
    night = "overnight" if float(transaction.get("hour_of_day") or 0) < 6 else "daytime"
    return " ".join([*signal_names, f"amount_{bucket}", f"product_{product}", night])
