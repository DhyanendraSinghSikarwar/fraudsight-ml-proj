# ADR 0004 — A hand-written state machine by default, LangGraph behind an adapter

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** who owns the triage control flow

## Context

The triage flow is:

```
redact → gather → retrieve → reason → validate → (repair → validate)* → end
```

Six nodes, one bounded cycle, no branching beyond the repair condition. The
default choice for an agent in 2026 is LangGraph.

## Decision

The default runner is ~80 lines in `triage/graph.py`.
`triage/langgraph_adapter.py` compiles the **same node functions** into a real
`StateGraph` when the `agent` extra is installed, and
`tests/integration/test_langgraph_parity.py` asserts both produce matching
briefs.

## Consequences

The escalation ladder in the playbook this project was built from says to add
orchestration only when a simpler thing demonstrably fails. Applied honestly to
this flow:

| What LangGraph earns its weight on | Present here? |
|---|---|
| Cyclic planning with dynamic routing | no — one fixed DAG |
| Interrupt / resume across process restarts | no — a triage run is seconds |
| Multi-agent handoff | no — one agent |
| Durable checkpointing | partially — we checkpoint to disk per node |
| Streaming intermediate state to a UI | no — the console renders the final brief |

What is left is state management, which `TriageState` does in a Pydantic model,
and retry, which the router already owns because it has to handle provider
failover anyway.

Three concrete costs of the framework we avoided:

1. **Cold start.** The triage path has a 6s p95 budget. Import and graph
   compilation cost is real against a budget that tight.
2. **`make demo` with zero services.** Every optional dependency is a reviewer
   who does not run the project.
3. **Debuggability.** When a triage produces a strange brief, the failure is in
   80 lines of visible control flow rather than inside a framework's executor.

The adapter exists so this is a reversible decision rather than a bet, and so
the claim "we evaluated LangGraph" is backed by working code instead of an
opinion.

## Alternatives rejected

**LangGraph as the default.** The honest answer to "why not?" is that it would
be resume-driven rather than requirement-driven. Reach for it when the
requirement below appears.

**Fold `retrieve` into `gather`.** Tempting for latency, but retrieval's query
is built from the signals derived from what `gather` returned, so it is a
genuine dependency, not an ordering preference.

**Let the model choose the node order.** With four tools that are all cheap and
all relevant, a planning call would add latency and a failure mode to save
nothing.

## When this flips

The moment triage needs **durable human-in-the-loop interrupts** — a case that
pauses mid-run waiting for an analyst to supply a document and resumes in a
different process an hour later. The built-in runner would need to grow a
checkpointer, a resume protocol, and an interrupt primitive, which is precisely
LangGraph's job. Flip the default to the adapter at that point; the node
functions do not change.
