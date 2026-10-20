"""Customer history tool.

The description string below is a prompt, not documentation. It is what the
model reads when deciding whether this tool is relevant, and vague tool
descriptions are the most common cause of wrong tool calls -- so it states the
key, the window, and the limitation rather than saying "gets customer info".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fraudsight.data.profiles import ProfileTables
from fraudsight.features.spec import _norm_key
from fraudsight.tools.base import ToolResult, ToolSpec

DESCRIPTION = (
    "Historical behaviour for the card on this transaction, keyed on card1. Returns "
    "transaction count, tenure, amount distribution (mean/median/p95/max), distinct "
    "billing addresses, distinct devices, and previously confirmed fraud on the card. "
    "Covers the training window only and is empty for a card never seen before -- an "
    "empty result is itself evidence, not an error."
)


def make(tables: ProfileTables) -> ToolSpec:
    async def run(scope: Mapping[str, Any]) -> ToolResult:
        key = _norm_key(scope.get("card1"))
        record = tables.customers.get(key)
        if record is None:
            return ToolResult(
                tool="customer_history",
                found=False,
                fields={
                    "customer_key": key,
                    "reason": "no prior transactions for this card in the reference window",
                },
                source="artifacts/profiles/customers.json",
            )
        return ToolResult(
            tool="customer_history",
            found=True,
            fields=dict(record),
            source="artifacts/profiles/customers.json",
        )

    return ToolSpec(
        name="customer_history",
        description=DESCRIPTION,
        fn=run,
        scope_keys=("card1",),
    )
