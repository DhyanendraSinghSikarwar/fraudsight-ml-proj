"""FastAPI service: /score, /triage, /decision, /health, /metrics.

`/score` and `/triage` are separate endpoints rather than one endpoint that
sometimes takes six seconds. They have different SLAs (50ms p99 vs 6s p95),
different failure modes, and different cost profiles, and a caller that needs a
scoring decision on the authorisation path must not be able to accidentally
block on an LLM call. `/score` returns the band; a caller who gets `triage` asks
for it explicitly.

`/decision` is the feedback loop, and it is the endpoint most portfolio projects
omit. An analyst's confirmed outcome is worth more than any amount of offline
tuning: it goes to the eval set and to the retraining data, and the counts are
on `/health` so "is the loop actually closing?" is a question with an answer.

Nothing here holds a transaction hostage to the LLM. If the provider is down,
`/triage` returns a score-only decision with `degraded: true` and a 200 -- the
NFR is that no transaction is ever dropped, and a 503 would drop it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudsight.config import settings
from fraudsight.observability.tracing import Tracer
from fraudsight.scoring.serve import Band
from fraudsight.triage.pipeline import TriagePipeline

app = FastAPI(
    title="fraudsight",
    version="0.1.0",
    description="GBDT in the hot path, LLM in the investigation layer.",
)

_state: dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    """Load artifacts once. Loading per request would put a 200ms model read
    inside a 50ms budget."""
    _state["pipeline"] = TriagePipeline.build()
    _state["tracer"] = Tracer.build()
    _state["started"] = datetime.now(UTC)
    _state["decisions_path"] = settings.artifact_dir / "analyst_decisions.jsonl"


def _pipeline() -> TriagePipeline:
    p = _state.get("pipeline")
    if p is None:  # pragma: no cover - startup guard
        raise HTTPException(503, "pipeline not initialised")
    return p


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    band: str
    action: str
    latency_ms: float
    bands: dict[str, float]


class TriageResponse(BaseModel):
    transaction_id: str
    score: float
    band: str
    action: str
    brief: dict[str, Any] | None = None
    outcome: str | None = None
    repairs: int = 0
    degraded: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace_id: str | None = None


class AnalystDecision(BaseModel):
    """What a human concluded. The `edited_*` fields capture the delta between
    the agent's brief and the analyst's call, which is the highest-signal
    training data the system produces."""

    transaction_id: str
    analyst_id: str
    decision: str = Field(pattern="^(approve|block|request_step_up_auth)$")
    agreed_with_agent: bool
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    edited_narrative: str | None = None
    rejected_evidence: list[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.post("/score", response_model=ScoreResponse)
def score(transaction: dict[str, Any] = Body(...)) -> ScoreResponse:
    """Hot path. Features + GBDT + calibration + band. No LLM, ever."""
    pipe = _pipeline()
    feats = pipe.features.compute(transaction)
    result = pipe.scorer.score_row(feats.model_row, feats.transaction_id)
    return ScoreResponse(
        transaction_id=result.transaction_id,
        score=round(result.score, 6),
        band=str(result.band),
        action={
            Band.AUTO_APPROVE: "approve",
            Band.AUTO_BLOCK: "block",
            Band.TRIAGE: "queue_for_triage",
        }[result.band],
        latency_ms=result.latency_ms,
        bands={"auto_approve_below": result.lo, "auto_block_above": result.hi},
    )


@app.post("/triage", response_model=TriageResponse)
async def triage(transaction: dict[str, Any] = Body(...)) -> TriageResponse:
    """Investigation path. Returns a brief, or a degraded score-only decision."""
    pipe = _pipeline()
    decision = await pipe.process(transaction)
    trace_id = None
    if decision.state is not None:
        trace_id = _state["tracer"].record(decision.state).get("trace_id")

    return TriageResponse(
        transaction_id=decision.transaction_id,
        score=round(decision.score, 6),
        band=str(decision.band),
        action=decision.action,
        brief=(
            decision.state.brief.model_dump(mode="json")
            if decision.state and decision.state.brief
            else None
        ),
        outcome=str(decision.state.outcome) if decision.state else None,
        repairs=decision.state.repairs if decision.state else 0,
        degraded=decision.state.degraded if decision.state else False,
        cost_usd=decision.cost_usd,
        latency_ms=decision.total_ms,
        trace_id=trace_id,
    )


@app.post("/decision")
def decision(payload: AnalystDecision) -> dict[str, Any]:
    """Capture an analyst's call. Feeds the eval set and the retraining data."""
    path: Path = _state["decisions_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **payload.model_dump()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return {"recorded": True, "transaction_id": payload.transaction_id}


@app.get("/health")
def health() -> dict[str, Any]:
    pipe = _pipeline()
    path: Path = _state["decisions_path"]
    captured = (
        sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if path.exists()
        else 0
    )
    return {
        "status": "ok",
        "started_utc": _state["started"].isoformat(timespec="seconds"),
        "bands": {"lo": pipe.scorer.lo, "hi": pipe.scorer.hi},
        "providers": pipe.graph.ctx.router.health(),
        "primary_provider": pipe.graph.ctx.router.primary.provider_name,
        "tools": pipe.graph.ctx.tools.names,
        "spend_today_usd": round(pipe.ledger.spent_today(), 6),
        "budget_remaining_usd": round(pipe.ledger.remaining_today(), 6),
        "tracing": _state["tracer"].health(),
        "analyst_decisions_captured": captured,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Funnel, latency and cost for everything this process has handled."""
    return _pipeline().summary()


@app.get("/tools")
def tools() -> list[dict[str, str]]:
    """The agent's allow-list, with the descriptions the model actually reads."""
    return _pipeline().graph.ctx.tools.describe()
