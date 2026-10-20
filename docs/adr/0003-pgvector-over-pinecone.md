# ADR 0003 — Local index by default, pgvector when it grows, no dedicated vector DB

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** where similar-case retrieval lives

## Context

The triage agent retrieves comparable resolved cases. The corpus is 1,500
records of about 60 words each. The default 2026 instinct is to reach for a
managed vector database.

## Decision

Three tiers, chosen by size rather than by fashion:

1. **Local** (default) — a numpy matrix plus an in-process BM25 index, persisted
   as `.npy` + `.jsonl`. 1,500 × 256 float32 is 1.5 MB.
2. **pgvector** — the same index in Postgres with an HNSW index, for when the
   corpus outgrows memory or more than one worker needs it.
3. **A dedicated vector database** — not implemented, and deliberately so.

## Consequences

The local default is why `make demo` runs on a reviewer's laptop with no
services. That is not a convenience; it is the difference between a reviewer who
runs the project and one who reads the README.

Retrieval is **hybrid** — dense plus BM25, fused with Reciprocal Rank Fusion —
and the reason is specific to this corpus. Every case narrative is a short
paragraph about a card, an amount and a device, so the embeddings crowd into a
narrow cone and dense-only top-k is close to arbitrary. Signal vocabulary
(`velocity_spike`, `device_unseen`, `thin_customer_history`) is what actually
discriminates, and that is exactly what a lexical retriever is good at.

Both backends run the fusion in Python rather than in SQL. Pushing RRF into a
query would be faster and would make the two backends quietly different systems
with incomparable retrieval metrics.

## Alternatives rejected

**Pinecone / Qdrant / Weaviate.** At 1,500 vectors these solve a problem the
system does not have. The crossover is roughly 10M vectors or genuine
multi-tenant filtering — neither is close. Adding one now would buy an
operational dependency, a monthly bill, and a second place for the corpus to be
stale.

**Dense-only retrieval.** Rejected on the corpus-geometry argument above.

**Elasticsearch for both halves.** Reasonable and rejected only on the
dependency argument: it is a JVM service to run for a 1.5 MB index.

## When this flips

- Corpus past ~1M cases, or memory pressure on the worker → pgvector (already
  implemented; flip `VECTOR_BACKEND`).
- Past ~10M vectors, or per-tenant ACL filtering at query time → revisit a
  dedicated store, and expect to rewrite the fusion layer.
