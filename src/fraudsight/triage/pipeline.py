"""End-to-end path for one transaction: features -> score -> route -> triage.

This is the object the API, the replay harness, the eval runner, and the console
all share. Having one assembly point is what makes the eval numbers describe the
deployed system: if the eval runner built its own pipeline, "89% agreement"
would be a fact about the eval runner.

The routing decision is the system's thesis in eight lines. 87% of transactions
never reach an LLM. They are scored in under a millisecond by a model that costs
nothing to run, and the only reason the expensive path exists is the ~12% where
the score is genuinely ambiguous.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudsight.config import Settings, settings as default_settings
from fraudsight.data.profiles import ProfileTables
from fraudsight.features.online import OnlineFeatureService
from fraudsight.llm import build_ledger, build_router
from fraudsight.llm.router import LLMRouter
from fraudsight.llm.usage import CostLedger
from fraudsight.scoring.serve import Band, ScoreResult, Scorer
from fraudsight.tools import build_registry, build_vector_store
from fraudsight.tools.base import ToolRegistry
from fraudsight.triage import rules as _rules  # noqa: F401 - registers the rule-based brief
from fraudsight.triage.graph import TriageContext, TriageGraph
from fraudsight.triage.state import Outcome, TriageState


@dataclass
class Decision:
    """What the system did with one transaction, and what it cost."""

    transaction_id: str
    score: float
    band: Band
    action: str
    score_ms: float
    feature_ms: float
    triage_ms: float = 0.0
    cost_usd: float = 0.0
    state: TriageState | None = None

    @property
    def total_ms(self) -> float:
        return round(self.feature_ms + self.score_ms + self.triage_ms, 3)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "score": round(self.score, 6),
            "band": str(self.band),
            "action": self.action,
            "feature_ms": self.feature_ms,
            "score_ms": self.score_ms,
            "triage_ms": self.triage_ms,
            "total_ms": self.total_ms,
            "cost_usd": round(self.cost_usd, 8),
        }
        if self.state is not None and self.state.brief is not None:
            out["brief"] = self.state.brief.model_dump(mode="json")
            out["outcome"] = str(self.state.outcome)
            out["repairs"] = self.state.repairs
            out["degraded"] = self.state.degraded
        return out


@dataclass
class TriagePipeline:
    scorer: Scorer
    features: OnlineFeatureService
    graph: TriageGraph
    ledger: CostLedger
    cfg: Settings
    trace_path: Path | None = None
    decisions: list[Decision] = field(default_factory=list)

    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        cfg: Settings | None = None,
        *,
        router: LLMRouter | None = None,
        tools: ToolRegistry | None = None,
        profiles: ProfileTables | None = None,
        write_traces: bool = True,
    ) -> TriagePipeline:
        cfg = cfg or default_settings
        cfg.ensure_dirs()

        scorer = Scorer.load(cfg.artifact_dir)
        profiles = profiles or ProfileTables.load(cfg.artifact_dir / "profiles")
        ledger = build_ledger(cfg)
        router = router or build_router(cfg, ledger=ledger)

        if tools is None:
            # The embedding call goes through the router, so a configured
            # embedding provider is used when present and the deterministic
            # hash embedder when not -- the retrieval mode is recorded either
            # way and shows up in the eval report.
            async def embed(texts: list[str]) -> tuple[list[list[float]], Any]:
                return await router.embed(texts, operation="retrieval_embed")

            tools = build_registry(
                profiles, store=build_vector_store(cfg), embed_fn=embed, cfg=cfg
            )

        ctx = TriageContext.build(router, tools, profiles, cfg)
        return cls(
            scorer=scorer,
            features=OnlineFeatureService(scorer.stats),
            graph=TriageGraph(ctx),
            ledger=ledger,
            cfg=cfg,
            trace_path=(cfg.trace_dir / "triage.jsonl") if write_traces else None,
        )

    # ------------------------------------------------------------------

    async def process(self, raw: Mapping[str, Any]) -> Decision:
        feats = self.features.compute(raw)
        result = self.scorer.score_row(feats.model_row, feats.transaction_id)

        decision = Decision(
            transaction_id=feats.transaction_id,
            score=result.score,
            band=result.band,
            action=_action_for(result),
            score_ms=result.latency_ms,
            feature_ms=feats.compute_ms,
        )

        if result.band is not Band.TRIAGE:
            self.decisions.append(decision)
            return decision

        # Only the ambiguous band pays for explanation and an LLM call.
        t0 = time.perf_counter()
        # Derived features first, raw values second -- raw wins on conflict.
        #
        # The order matters more than it looks. `model_row` holds *encoded*
        # categoricals: any level outside the pinned vocabulary has already
        # collapsed to `__other__`. Letting those overwrite the raw record
        # replaced the human-readable device string, merchant code and email
        # domain with encoder sentinels, so the agent reasoned about
        # `__other__`, the analyst console displayed `__other__`, and -- worst
        # -- attacker-controlled free text never reached the injection scanner
        # because the encoder had already eaten it. The scanner then reported a
        # clean run on a poisoned transaction.
        enriched = {**feats.model_row, **dict(raw)}
        enriched["__velocity__"] = feats.velocity_summary()
        enriched["__explanation__"] = self.scorer.explain_row(feats.model_row)

        state = await self.graph.run(enriched, score=result.score, band=str(result.band))
        decision.triage_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        decision.cost_usd = state.cost_usd
        decision.state = state
        decision.action = _action_for_brief(state)

        self._write_trace(state)
        self.decisions.append(decision)
        return decision

    # ------------------------------------------------------------------

    def _write_trace(self, state: TriageState) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # The timestamp is what lets the daily report aggregate one day rather
        # than the entire append-only history. Without it the report silently
        # mixed every run since the file was created and attributed months-old
        # policy failures to today.
        row = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **state.trace()}
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def summary(self) -> dict[str, Any]:
        """Funnel + latency + cost across everything processed so far."""
        n = len(self.decisions) or 1
        bands: dict[str, int] = {}
        for d in self.decisions:
            bands[str(d.band)] = bands.get(str(d.band), 0) + 1
        triaged = [d for d in self.decisions if d.band is Band.TRIAGE]
        cost = sum(d.cost_usd for d in self.decisions)
        return {
            "transactions": len(self.decisions),
            "band_counts": bands,
            "band_rates": {k: round(v / n, 6) for k, v in bands.items()},
            "triaged": len(triaged),
            "scoring_latency": self.scorer.latency.percentiles(),
            "triage_latency": _percentiles([d.triage_ms for d in triaged]),
            "cost_usd_total": round(cost, 6),
            "cost_per_triage_usd": round(cost / max(len(triaged), 1), 8),
            "cost_per_1k_transactions_usd": round(1000.0 * cost / n, 6),
            "spend_today_usd": round(self.ledger.spent_today(), 6),
            "provider_health": self.graph.ctx.router.health(),
        }


def _action_for(result: ScoreResult) -> str:
    return {
        Band.AUTO_APPROVE: "approved",
        Band.AUTO_BLOCK: "blocked",
        Band.TRIAGE: "queued_for_triage",
    }[result.band]


def _action_for_brief(state: TriageState) -> str:
    """Post-triage routing.

    High-confidence briefs close the case without a human; everything else lands
    in the analyst queue. The confidence floors live in guardrails/output.py so
    the policy is enforced in one place rather than re-derived here.
    """
    if state.brief is None or state.outcome is not Outcome.COMPLETED:
        return "escalated_to_analyst"
    rec = str(state.brief.recommendation)
    if rec == "approve":
        return "approved_after_triage"
    if rec == "block":
        return "blocked_after_triage"
    if rec == "request_step_up_auth":
        return "step_up_requested"
    return "escalated_to_analyst"


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    import numpy as np

    arr = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(arr[-1]), 3),
    }
