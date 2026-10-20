"""Merchant profile tool.

IEEE-CIS has no merchant column. `ProductCD x addr1` is the closest available
proxy and the tool says so in every result, including the `basis` field, so the
agent's brief inherits the caveat instead of laundering a proxy into a fact.
Naming the approximation in the payload rather than only in a docstring is
deliberate: the model reads the payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fraudsight.data.profiles import ProfileTables, merchant_key
from fraudsight.tools.base import ToolResult, ToolSpec

DESCRIPTION = (
    "Risk profile for the merchant this transaction was made at. IEEE-CIS ships no "
    "merchant identifier, so the key is a PROXY: product code x billing region. Returns "
    "transaction volume, amount distribution, distinct cards, and the merchant's fraud "
    "rate as a multiple of the portfolio average. Because the key is a proxy, treat a "
    "high fraud multiple as a segment-level signal, not as a claim about a named business."
)


def make(tables: ProfileTables) -> ToolSpec:
    async def run(scope: Mapping[str, Any]) -> ToolResult:
        key = merchant_key(scope.get("ProductCD"), scope.get("addr1"))
        record = tables.merchants.get(key)
        if record is None:
            return ToolResult(
                tool="merchant_profile",
                found=False,
                fields={
                    "merchant_key": key,
                    "reason": "no transactions for this product/region pair in the reference window",
                },
                source="artifacts/profiles/merchants.json",
            )
        return ToolResult(
            tool="merchant_profile",
            found=True,
            fields=dict(record),
            source="artifacts/profiles/merchants.json",
        )

    return ToolSpec(
        name="merchant_profile",
        description=DESCRIPTION,
        fn=run,
        scope_keys=("ProductCD",),
    )
