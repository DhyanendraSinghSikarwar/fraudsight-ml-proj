"""Compile the same triage nodes into a real LangGraph `StateGraph`.

Available when the `agent` extra is installed. Same nodes, same order, same
bounded repair cycle -- the only thing that changes is who owns the control flow
and, with a `PostgresSaver`, where the checkpoints live.

This exists for the case the built-in runner does not cover: durable
human-in-the-loop interrupts. If a triage case has to pause mid-run, wait for an
analyst to supply a missing document, and resume in a different process an hour
later, LangGraph's checkpointer handles it and the built-in runner would need to
grow one. Until that requirement is real, `graph.TriageGraph` is the default --
see docs/adr/0004-no-agent-framework.md.

Keeping both paths honest is the job of
tests/integration/test_langgraph_parity.py, which runs the same fixture through
both and asserts the resulting briefs match.
"""

from __future__ import annotations

from typing import Any

from fraudsight.triage.graph import (
    TriageContext,
    gather_evidence,
    produce_brief,
    redact_pii,
    repair_brief,
    retrieve_similar,
    validate_brief,
)
from fraudsight.triage.state import TriageState


class LangGraphUnavailable(RuntimeError):
    pass


def build(ctx: TriageContext, *, checkpointer: Any | None = None) -> Any:
    """Return a compiled LangGraph app with the same semantics as TriageGraph."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - optional extra
        raise LangGraphUnavailable(
            "langgraph not installed. `uv pip install -e '.[agent]'`, or use the "
            "built-in runner in triage/graph.py (the default)."
        ) from exc

    g = StateGraph(TriageState)

    # The nodes take (ctx, state); LangGraph passes only state. Binding ctx here
    # rather than making it a graph input keeps the node signatures identical
    # between the two runners, which is what lets the parity test be meaningful.
    def bind(fn: Any) -> Any:
        async def node(state: TriageState) -> TriageState:
            return await fn(ctx, state)

        node.__name__ = fn.__name__
        return node

    g.add_node("redact", bind(redact_pii))
    g.add_node("gather", bind(gather_evidence))
    g.add_node("retrieve", bind(retrieve_similar))
    g.add_node("reason", bind(produce_brief))
    g.add_node("validate", bind(validate_brief))
    g.add_node("repair", bind(repair_brief))

    g.set_entry_point("redact")
    g.add_edge("redact", "gather")
    g.add_edge("gather", "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", "validate")
    g.add_conditional_edges(
        "validate",
        lambda s: (
            "repair"
            if s.validation_error and s.repairs < ctx.cfg.triage_max_repairs
            else END
        ),
        {"repair": "repair", END: END},
    )
    g.add_edge("repair", "validate")

    return g.compile(checkpointer=checkpointer)


def postgres_checkpointer(dsn: str) -> Any:  # pragma: no cover - needs a database
    """Durable checkpoints, for the resume-across-processes case that is the
    whole reason to reach for this adapter."""
    from langgraph.checkpoint.postgres import PostgresSaver

    return PostgresSaver.from_conn_string(dsn)
