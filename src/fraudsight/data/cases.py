"""Two labelled corpora, built from disjoint time windows.

**The case corpus** (`build_corpus`) is retrieval memory: resolved historical
cases the triage agent can look up for base rates. Built from the *training*
window only, because a case for a val-window transaction that could retrieve its
own neighbours from the val window would leak, and would leak invisibly -- every
Track B metric would improve and nothing would look wrong.

**The eval set** (`build_eval_cases`) is the regression suite: 150 transactions
with known outcomes, drawn from the *val* window. Built before the agent existed,
which is the discipline the whole eval section rests on -- an eval set written
after the agent is an eval set shaped by the agent's behaviour.

The three strata come from the blueprint: 50 confirmed fraud, 50 confirmed
legitimate, 50 genuinely ambiguous. We have no record of analysts disagreeing, so
"ambiguous" is defined mechanically and the definition is published rather than
implied: transactions inside the triage band where **the label contradicts the
score's direction** -- the model leaned one way and reality went the other. Those
are the cases where a human's judgement has to add something, and they are
exactly where a rule engine and an LLM should diverge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudsight.config import settings
from fraudsight.data.profiles import ProfileTables, merchant_key, portfolio_baseline
from fraudsight.features.spec import LABEL, TIME, TXN_ID, _norm_key
from fraudsight.llm.deterministic import _hash_embed
from fraudsight.tools.base import ToolResult
from fraudsight.tools.similar_cases import build_query
from fraudsight.tools.vectorstore import CaseRecord, LocalVectorStore
from fraudsight.triage.evidence import EvidenceBundle, derive_signals

CORPUS_SIZE = 1_500
EVAL_PER_STRATUM = 50
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class EvalCase:
    """One labelled triage case. `expected_action` is what an analyst working
    from a complete picture should have done, derived from the confirmed
    outcome."""

    case_id: str
    transaction_id: str
    stratum: str  # confirmed_fraud | confirmed_legitimate | ambiguous
    label: int
    model_score: float
    expected_action: str  # block | approve
    day: float
    amount: float
    transaction: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Retrieval corpus
# --------------------------------------------------------------------------


def build_corpus(
    profiles: ProfileTables,
    *,
    size: int = CORPUS_SIZE,
    seed: int = 17,
) -> LocalVectorStore:
    """Sample resolved training-window cases, narrate them, embed them.

    Sampling is stratified to roughly 40% fraud. The training window is 3.4%
    fraud, and a corpus that mirrored that would return five legitimate
    neighbours for every query and be useless as a base rate for the fraud
    patterns. The over-sampling is deliberate and is why `similar_cases`
    reports `confirmed_fraud_among_returned` rather than letting the agent read
    the mix as a population rate.
    """
    path = settings.data_dir / "processed" / "train.parquet"
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)

    fraud_idx = np.flatnonzero(df[LABEL].to_numpy() == 1)
    legit_idx = np.flatnonzero(df[LABEL].to_numpy() == 0)
    n_fraud = min(int(size * 0.4), fraud_idx.size)
    n_legit = min(size - n_fraud, legit_idx.size)
    picked = np.concatenate(
        [rng.choice(fraud_idx, n_fraud, replace=False), rng.choice(legit_idx, n_legit, replace=False)]
    )
    rng.shuffle(picked)

    baseline = portfolio_baseline(profiles)
    records: list[CaseRecord] = []
    for i in picked:
        row = df.iloc[int(i)].to_dict()
        bundle = _offline_bundle(row, profiles, baseline)
        signals = derive_signals(bundle)
        outcome = "confirmed_fraud" if int(row[LABEL]) == 1 else "confirmed_legitimate"
        records.append(
            CaseRecord(
                case_id=f"case_{int(row[TXN_ID])}",
                narrative=_case_narrative(bundle, signals, outcome),
                signals=[s.name for s in signals],
                outcome=outcome,
                metadata={
                    "amount": round(float(row.get("TransactionAmt") or 0.0), 2),
                    "day": round(float(row[TIME]) / SECONDS_PER_DAY, 2),
                    "product": str(row.get("ProductCD")),
                    "resolved": True,
                },
            )
        )

    vectors = np.array(
        [_hash_embed(r.text(), settings.embedding_dim) for r in records], dtype=np.float32
    )
    store = LocalVectorStore(records, vectors)
    store.save(settings.artifact_dir / "cases")
    return store


def _case_narrative(bundle: EvidenceBundle, signals: list[Any], outcome: str) -> str:
    """A resolved case reads as a short retrospective.

    Written from the same `Signal.detail` strings the live evidence layer
    produces, so a query and a corpus entry share vocabulary. Retrieval quality
    depends far more on that shared vocabulary than on the prose.
    """
    txn = bundle.transaction
    top = [s for s in signals if s.direction == "risk_increasing"][:3]
    mitigating = [s for s in signals if s.direction == "risk_decreasing"][:2]
    verdict = (
        "Resolved as confirmed fraud."
        if outcome == "confirmed_fraud"
        else "Resolved as legitimate; the alert was a false positive."
    )
    parts = [
        f"${txn.get('amount', 0):,.2f} {txn.get('product_code', 'unknown')} authorisation "
        f"at {int(txn.get('hour_of_day', 0)):02d}:00.",
        *[s.detail for s in top],
        *(["Mitigating: " + " ".join(s.detail for s in mitigating)] if mitigating else []),
        verdict,
    ]
    return " ".join(parts)


# --------------------------------------------------------------------------
# Eval set
# --------------------------------------------------------------------------


def build_eval_cases(
    *,
    lo: float,
    hi: float,
    per_stratum: int = EVAL_PER_STRATUM,
    seed: int = 23,
) -> list[EvalCase]:
    """150 cases from the val window, inside the triage band.

    Only band-eligible transactions are used. Evaluating the agent on
    transactions the system would have auto-approved measures something the
    system never does, and inflates agreement because the easy cases dominate.
    """
    from fraudsight.scoring.dataset import load_split, load_stats
    from fraudsight.scoring.serve import Scorer

    stats = load_stats()
    split = load_split("val", stats)
    scorer = Scorer.load()
    probs = scorer.score_frame(split.X)

    in_band = (probs >= lo) & (probs < hi)
    labels = split.y
    rng = np.random.default_rng(seed)

    # "Ambiguous" = inside the band and the label contradicts where the score
    # leans, using the band midpoint as the lean.
    mid = (lo + hi) / 2.0
    leans_risky = probs >= mid
    contradicted = in_band & (leans_risky != (labels == 1))

    fraud_pool = np.flatnonzero(in_band & (labels == 1) & ~contradicted)
    legit_pool = np.flatnonzero(in_band & (labels == 0) & ~contradicted)
    ambig_pool = np.flatnonzero(contradicted)

    def take(pool: np.ndarray, n: int) -> np.ndarray:
        if pool.size == 0:
            return pool
        return rng.choice(pool, min(n, pool.size), replace=False)

    chosen = [
        ("confirmed_fraud", take(fraud_pool, per_stratum)),
        ("confirmed_legitimate", take(legit_pool, per_stratum)),
        ("ambiguous", take(ambig_pool, per_stratum)),
    ]

    cases: list[EvalCase] = []
    for stratum, idx in chosen:
        for i in idx:
            i = int(i)
            row = split.X.iloc[i].to_dict()
            row[TXN_ID] = int(split.txn_id[i])
            row[TIME] = int(split.time[i])
            label = int(labels[i])
            cases.append(
                EvalCase(
                    case_id=f"eval_{int(split.txn_id[i])}",
                    transaction_id=str(int(split.txn_id[i])),
                    stratum=stratum,
                    label=label,
                    model_score=round(float(probs[i]), 6),
                    expected_action="block" if label == 1 else "approve",
                    day=round(float(split.time[i]) / SECONDS_PER_DAY, 2),
                    amount=round(float(row.get("TransactionAmt") or 0.0), 2),
                    transaction=_jsonable(row),
                )
            )
    return cases


def save_eval_cases(cases: list[EvalCase], path: Path | None = None) -> Path:
    p = path or (Path("evals") / "datasets" / "triage_holdout.jsonl")
    p = p if p.is_absolute() else (settings.artifact_dir.parent / p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c.as_dict(), default=str) + "\n")
    return p


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    p = path or (settings.artifact_dir.parent / "evals" / "datasets" / "triage_holdout.jsonl")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `fraudsight evalset`")
    return [
        EvalCase(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# Offline bundle (no tool registry, no async -- corpus build only)
# --------------------------------------------------------------------------


def _offline_bundle(
    row: dict[str, Any], profiles: ProfileTables, baseline: dict[str, float]
) -> EvidenceBundle:
    """Assemble the same evidence shape the live path builds, synchronously.

    Reusing `derive_signals` here rather than writing a second narrative
    generator is the point: the corpus and the live queries share one signal
    vocabulary, which is what makes lexical retrieval work at all.
    """
    from fraudsight.triage.graph import _transaction_view

    card = _norm_key(row.get("card1"))
    device = _norm_key(row.get("DeviceInfo"))
    mkey = merchant_key(row.get("ProductCD"), row.get("addr1"))

    tools: dict[str, ToolResult] = {}
    for name, table, key, src in (
        ("customer_history", profiles.customers, card, "profiles/customers.json"),
        ("device_intel", profiles.devices, device, "profiles/devices.json"),
        ("merchant_profile", profiles.merchants, mkey, "profiles/merchants.json"),
    ):
        rec = table.get(key)
        tools[name] = ToolResult(
            tool=name, found=rec is not None, fields=dict(rec or {}), source=src
        )

    row = {**row, TXN_ID: row.get(TXN_ID, 0)}
    return EvidenceBundle(
        transaction_id=str(row.get(TXN_ID)),
        transaction=_transaction_view(row),
        model={"score": 0.0, "band": "corpus", "top_contributions": []},
        tools=tools,
        velocity={},
        portfolio=baseline,
        missing=[n for n, r in tools.items() if not r.found],
    )


def corpus_query_for(case: CaseRecord) -> str:
    return build_query(case.signals, {"amount": case.metadata.get("amount", 0.0)})


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = round(float(v), 6)
        else:
            out[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
    return out
