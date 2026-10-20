"""Similar-case retrieval: hybrid lexical + dense, with a reranking pass.

Two backends behind one interface. `local` is a numpy matrix on disk and is the
default because it makes the whole system runnable with no services; `pgvector`
is the same index in Postgres for when the corpus outgrows memory. The retrieval
*metrics* in the eval report are computed identically for both, so the choice is
a deployment decision rather than a quality one.

Retrieval here is hybrid on purpose. Pure dense retrieval over fraud case
narratives collapses badly: every case is 60 words about a card, an amount, and a
device, so the embeddings crowd into a narrow cone and the top-k is close to
arbitrary. Exact-match signal overlap (BM25-ish on the signal vocabulary) is what
actually distinguishes "velocity spike on a thin-history card" from "unseen
device on an established card", and the fusion of the two beats either alone --
measured in artifacts/retrieval_metrics.json rather than asserted.

Fusion is Reciprocal Rank Fusion rather than a weighted score sum: the two
scorers are on incomparable scales (cosine in [-1,1], BM25 unbounded), and
normalising them per-query is exactly the kind of quiet coupling that makes a
retrieval system unreproducible.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# RRF's rank offset, from the original paper. It is deliberately large relative
# to the ranks in play: 1/(60+1) and 1/(60+4) are close together, so no single
# retriever's top hit can carry a document on its own. A case has to place well
# in *both* rankings to surface -- which is what suppresses the lexical
# retriever's habit of adoring one rare shared token.
RRF_K = 60


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    narrative: str
    signals: list[str]
    outcome: str  # confirmed_fraud | confirmed_legitimate
    metadata: dict[str, Any]

    def text(self) -> str:
        return f"{self.narrative} signals: {' '.join(self.signals)}"


@dataclass(frozen=True)
class Retrieved:
    case: CaseRecord
    score: float
    dense_rank: int | None
    lexical_rank: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "outcome": self.case.outcome,
            "narrative": self.case.narrative,
            "signals": self.case.signals,
            "fusion_score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            **{k: v for k, v in self.case.metadata.items() if k in ("amount", "day", "product")},
        }


class BM25:
    """Okapi BM25 over the case corpus. ~200 lines of scikit would do this, but
    the corpus is small and a dependency-free implementation keeps `make demo`
    honest about having no services."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: list[Counter[str]] = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query: list[str]) -> NDArray[np.float64]:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n:
            return out
        for i, tf in enumerate(self.tf):
            length = self.lengths[i] or 1
            total = 0.0
            for term in query:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                total += idf * f * (self.k1 + 1) / denom
            out[i] = total
        return out


class LocalVectorStore:
    """Dense matrix + BM25 index, persisted as .npy + .jsonl."""

    def __init__(self, cases: list[CaseRecord], vectors: NDArray[np.float64]) -> None:
        if len(cases) != vectors.shape[0]:
            raise ValueError(f"{len(cases)} cases but {vectors.shape[0]} vectors")
        self.cases = cases
        self.vectors = vectors.astype(np.float32)
        self._bm25 = BM25([_tokens(c.text()) for c in cases])

    def __len__(self) -> int:
        return len(self.cases)

    # -- persistence ---------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "case_vectors.npy", self.vectors)
        with (directory / "cases.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.cases:
                fh.write(json.dumps(c.__dict__) + "\n")

    @classmethod
    def load(cls, directory: Path) -> LocalVectorStore:
        vec_path = directory / "case_vectors.npy"
        case_path = directory / "cases.jsonl"
        if not vec_path.exists() or not case_path.exists():
            raise FileNotFoundError(
                f"case index missing in {directory} -- run `fraudsight cases`"
            )
        cases = [
            CaseRecord(**json.loads(line))
            for line in case_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(cases, np.load(vec_path))

    # -- search --------------------------------------------------------

    def search(
        self,
        query_vector: NDArray[np.float64] | None,
        query_text: str,
        *,
        top_k: int = 5,
        candidates: int = 50,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[Retrieved]:
        if not self.cases:
            return []

        keep = np.array([c.case_id not in exclude_ids for c in self.cases])
        idx_pool = np.flatnonzero(keep)
        if idx_pool.size == 0:
            return []

        dense_rank: dict[int, int] = {}
        if query_vector is not None and self.vectors.size:
            q = np.asarray(query_vector, dtype=np.float32)
            sims = self.vectors[idx_pool] @ q
            order = idx_pool[np.argsort(-sims)][:candidates]
            dense_rank = {int(i): r for r, i in enumerate(order)}

        lex = self._bm25.scores(_tokens(query_text))
        lex_masked = np.where(keep, lex, -np.inf)
        lex_order = np.argsort(-lex_masked)[:candidates]
        lexical_rank = {int(i): r for r, i in enumerate(lex_order) if np.isfinite(lex_masked[i])}

        fused: dict[int, float] = {}
        for i, r in dense_rank.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        for i, r in lexical_rank.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Retrieved(
                case=self.cases[i],
                score=s,
                dense_rank=dense_rank.get(i),
                lexical_rank=lexical_rank.get(i),
            )
            for i, s in ranked
        ]


class PgVectorStore:
    """Postgres + pgvector. Same search contract, HNSW index.

    Kept deliberately thin: the fusion logic lives in Python so both backends
    rank identically and the eval numbers are comparable. Pushing RRF into SQL
    would be faster and would make the two backends quietly different systems.
    """

    def __init__(self, dsn: str, *, dim: int, table: str = "fraud_cases") -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise RuntimeError("psycopg not installed. `uv pip install -e '.[db]'`") from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.dim = dim
        self.table = table

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    case_id   TEXT PRIMARY KEY,
                    narrative TEXT NOT NULL,
                    signals   TEXT[] NOT NULL,
                    outcome   TEXT NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dim}) NOT NULL,
                    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', narrative)) STORED
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_tsv ON {self.table} USING gin (tsv)"
            )
            conn.commit()

    def upsert(self, cases: list[CaseRecord], vectors: NDArray[np.float64]) -> int:
        rows = [
            (c.case_id, c.narrative, c.signals, c.outcome, json.dumps(c.metadata), list(map(float, v)))
            for c, v in zip(cases, vectors, strict=True)
        ]
        with self._psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table} (case_id, narrative, signals, outcome, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (case_id) DO UPDATE SET
                    narrative = EXCLUDED.narrative,
                    signals   = EXCLUDED.signals,
                    outcome   = EXCLUDED.outcome,
                    metadata  = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def search(
        self,
        query_vector: NDArray[np.float64] | None,
        query_text: str,
        *,
        top_k: int = 5,
        candidates: int = 50,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[Retrieved]:
        vec = list(map(float, query_vector)) if query_vector is not None else [0.0] * self.dim
        excl = list(exclude_ids) or [""]
        with self._psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT case_id, narrative, signals, outcome, metadata,
                       embedding <=> %s::vector AS dist
                FROM {self.table}
                WHERE NOT (case_id = ANY(%s))
                ORDER BY dist ASC LIMIT %s
                """,
                (vec, excl, candidates),
            )
            dense = cur.fetchall()
            cur.execute(
                f"""
                SELECT case_id, narrative, signals, outcome, metadata,
                       ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
                FROM {self.table}
                WHERE NOT (case_id = ANY(%s)) AND tsv @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC LIMIT %s
                """,
                (query_text, excl, query_text, candidates),
            )
            lexical = cur.fetchall()

        by_id: dict[str, CaseRecord] = {}
        dense_rank: dict[str, int] = {}
        lexical_rank: dict[str, int] = {}
        for r, row in enumerate(dense):
            by_id[row[0]] = _record(row)
            dense_rank[row[0]] = r
        for r, row in enumerate(lexical):
            by_id.setdefault(row[0], _record(row))
            lexical_rank[row[0]] = r

        fused: dict[str, float] = {}
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r + 1)
        for cid, r in lexical_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Retrieved(by_id[cid], s, dense_rank.get(cid), lexical_rank.get(cid))
            for cid, s in ranked
        ]


def _record(row: tuple[Any, ...]) -> CaseRecord:
    meta = row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}")
    return CaseRecord(
        case_id=row[0], narrative=row[1], signals=list(row[2]), outcome=row[3], metadata=meta
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def retrieval_metrics(
    results: list[list[Retrieved]], relevant: list[frozenset[str]], k: int = 5
) -> dict[str, float]:
    """precision@k, recall@k and MRR over a labelled query set.

    "Relevance" here is defined mechanically: a retrieved case is relevant if it
    shares the query's outcome AND at least one signal. Stating the definition
    matters more than the number -- a retrieval precision quoted without one is
    not a measurement.
    """
    if not results:
        return {}
    precisions, recalls, rr = [], [], []
    for got, rel in zip(results, relevant, strict=True):
        ids = [r.case.case_id for r in got[:k]]
        hits = [i for i in ids if i in rel]
        precisions.append(len(hits) / max(len(ids), 1))
        recalls.append(len(hits) / max(len(rel), 1) if rel else 0.0)
        first = next((n for n, i in enumerate(ids, 1) if i in rel), None)
        rr.append(1.0 / first if first else 0.0)
    return {
        f"precision_at_{k}": round(float(np.mean(precisions)), 4),
        f"recall_at_{k}": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(rr)), 4),
        "queries": len(results),
    }
