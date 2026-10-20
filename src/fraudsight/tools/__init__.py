"""Tool registry assembly.

One function builds the whole allow-list. Nothing else in the system is allowed
to construct a `ToolSpec`, which is what keeps "what can the agent reach?" a
question with a single, greppable answer.
"""

from __future__ import annotations

from typing import Any

from fraudsight.config import Settings, settings as default_settings
from fraudsight.data.profiles import ProfileTables
from fraudsight.tools import customer_history, device_intel, merchant_profile, similar_cases
from fraudsight.tools.base import ScopeViolation, ToolError, ToolRegistry, ToolResult, ToolSpec
from fraudsight.tools.vectorstore import (
    CaseRecord,
    LocalVectorStore,
    PgVectorStore,
    Retrieved,
    retrieval_metrics,
)

__all__ = [
    "CaseRecord",
    "LocalVectorStore",
    "PgVectorStore",
    "Retrieved",
    "ScopeViolation",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_registry",
    "build_vector_store",
    "retrieval_metrics",
]


def build_vector_store(cfg: Settings | None = None) -> Any:
    cfg = cfg or default_settings
    if cfg.vector_backend == "pgvector":
        store = PgVectorStore(cfg.database_url, dim=cfg.embedding_dim)
        store.ensure_schema()
        return store
    return LocalVectorStore.load(cfg.artifact_dir / "cases")


def build_registry(
    profiles: ProfileTables,
    *,
    store: Any | None = None,
    embed_fn: Any | None = None,
    cfg: Settings | None = None,
) -> ToolRegistry:
    cfg = cfg or default_settings
    registry = ToolRegistry(max_concurrency=cfg.max_tool_concurrency)
    registry.register(customer_history.make(profiles))
    registry.register(device_intel.make(profiles))
    registry.register(merchant_profile.make(profiles))
    registry.register(similar_cases.make(store or build_vector_store(cfg), embed_fn))
    return registry
