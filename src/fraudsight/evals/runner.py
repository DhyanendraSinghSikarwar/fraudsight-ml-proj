"""Run the triage agent over the eval set and produce the report.

The report header records the prompt hashes, provider, model, band thresholds
and dataset digest. That matters more than it sounds: an eval number without
those five facts cannot be compared to another eval number, and a README table
of scores from six different configurations is not a results table.

`--repeat` runs the whole suite N times and reports mean +- std. The default is
1 for iteration speed and 3 for anything published; the gate refuses to pass a
published run with fewer than 3 unless explicitly told to.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudsight.config import Settings, settings as default_settings
from fraudsight.data.cases import EvalCase, load_eval_cases
from fraudsight.evals import judges as judge_mod
from fraudsight.evals.metrics import CaseResult, compare_to_floor, summarise, variance_report
from fraudsight.guardrails.output import evidence_is_grounded
from fraudsight.llm import build_ledger, build_router
from fraudsight.triage import prompts
from fraudsight.triage.pipeline import TriagePipeline
from fraudsight.triage.state import Outcome


@dataclass
class RunConfig:
    provider: str | None = None
    model: str | None = None
    repeat: int = 1
    concurrency: int = 8
    judge: bool = True
    limit: int | None = None


async def run_once(
    cases: list[EvalCase],
    cfg: Settings,
    *,
    run_cfg: RunConfig,
) -> tuple[list[CaseResult], dict[str, Any]]:
    ledger = build_ledger(cfg)
    router = build_router(cfg, ledger=ledger, model=run_cfg.model)
    pipeline = TriagePipeline.build(cfg, router=router, write_traces=False)

    sem = asyncio.Semaphore(run_cfg.concurrency)
    judge = judge_mod.Judge(router) if run_cfg.judge else None

    async def one(case: EvalCase) -> CaseResult | None:
        async with sem:
            wall_start = time.perf_counter()
            decision = await pipeline.process(case.transaction)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
        state = decision.state
        if state is None:
            # The scorer routed it out of the triage band. That happens when the
            # eval set was built against different thresholds; it is a real
            # signal, not something to silently drop.
            return None

        brief = state.brief
        grounded, total = (
            evidence_is_grounded(brief, state.bundle) if brief else (0, 0)
        )
        judged: dict[str, Any] | None = None
        if judge is not None and brief is not None:
            rows = await judge.judge_many([(brief, state.bundle)])
            judged = rows[0]

        return CaseResult(
            case_id=case.case_id,
            stratum=case.stratum,
            label=case.label,
            expected_action=case.expected_action,
            recommendation=str(brief.recommendation) if brief else "escalate_to_analyst",
            confidence=float(brief.confidence) if brief else 0.0,
            schema_ok=brief is not None,
            first_attempt_ok=brief is not None and state.repairs == 0,
            repairs=state.repairs,
            policy_violations=[
                v.get("rule", "?") for v in state.policy.get("violations", [])
            ],
            grounded_claims=grounded,
            total_claims=total,
            evidence_count=len(brief.evidence) if brief else 0,
            has_mitigating=bool(brief and brief.has_mitigating_evidence()),
            missing_information=len(brief.missing_information) if brief else 0,
            # Service time = the sum of the node spans, not wall clock. Under
            # N-way concurrency the wall clock for one case includes waiting
            # behind the other N-1, so reporting it as latency would describe
            # the harness's queue rather than the system's response time.
            #
            # Caveat worth stating because it makes the number look worse than
            # the system is: the deterministic provider runs *in this process*,
            # so concurrent eval cases contend on the GIL and every span
            # inflates. Against a real provider the calls are IO-bound and
            # overlap properly. `fraudsight replay` measures latency with the
            # concurrency the service actually uses; this figure is comparative,
            # not an SLA claim.
            latency_ms=round(state.total_ms, 3),
            wall_ms=round(wall_ms, 3),
            cost_usd=state.cost_usd,
            degraded=state.degraded,
            outcome=str(state.outcome or Outcome.ESCALATED_ERROR),
            tools_found=sum(1 for r in state.bundle.tools.values() if r.found),
            similar_cases=len(brief.similar_cases) if brief else 0,
            judge=judged,
        )

    gathered = await asyncio.gather(*(one(c) for c in cases))
    results = [r for r in gathered if r is not None]
    meta = {
        "provider": router.primary.provider_name,
        "model": getattr(router.primary, "model", "unknown"),
        "fallbacks": [c.provider_name for c in router.fallbacks],
        "judge_provider": router.primary.provider_name if run_cfg.judge else None,
        "skipped_out_of_band": len(cases) - len(results),
        "spend_usd": round(ledger.spent_today(), 6),
    }
    return results, meta


async def run(
    cfg: Settings | None = None,
    *,
    run_cfg: RunConfig | None = None,
    dataset: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or default_settings
    run_cfg = run_cfg or RunConfig()
    cases = load_eval_cases(dataset)
    if run_cfg.limit:
        cases = cases[: run_cfg.limit]

    started = datetime.now(UTC)
    all_runs: list[dict[str, Any]] = []
    all_results: list[list[CaseResult]] = []
    meta: dict[str, Any] = {}

    for _ in range(max(1, run_cfg.repeat)):
        results, meta = await run_once(cases, cfg, run_cfg=run_cfg)
        all_results.append(results)
        all_runs.append(summarise(results))

    headline = all_runs[0] if len(all_runs) == 1 else _mean_of(all_runs)

    report: dict[str, Any] = {
        "header": _header(cfg, cases, meta, started, run_cfg),
        "metrics": headline,
        "variance": variance_report(all_runs) if len(all_runs) > 1 else None,
        "runs": all_runs if len(all_runs) > 1 else None,
        "per_case": [_case_row(r) for r in all_results[-1]],
    }
    return report


async def run_with_floor(
    cfg: Settings | None = None, *, run_cfg: RunConfig | None = None
) -> dict[str, Any]:
    """Agent vs the rule-based floor, in one report.

    Both halves run against the same cases, the same tools, and the same
    retrieval index -- only the provider differs. That is what makes the delta
    attributable to the model rather than to anything else that changed.
    """
    cfg = cfg or default_settings
    run_cfg = run_cfg or RunConfig()

    agent_report = await run(cfg, run_cfg=run_cfg)

    floor_cfg = cfg.model_copy(
        update={"primary_provider": "deterministic", "fallback_provider": None}
    )
    floor_report = await run(floor_cfg, run_cfg=run_cfg)

    agent_report["floor"] = {
        "header": floor_report["header"],
        "metrics": floor_report["metrics"],
    }
    agent_report["vs_floor"] = compare_to_floor(
        agent_report["metrics"], floor_report["metrics"]
    )
    return agent_report


def save(report: dict[str, Any], path: Path | None = None) -> Path:
    p = path or (default_settings.artifact_dir / "eval_report.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p


# --------------------------------------------------------------------------


def _header(
    cfg: Settings,
    cases: list[EvalCase],
    meta: dict[str, Any],
    started: datetime,
    run_cfg: RunConfig,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        "".join(sorted(c.case_id for c in cases)).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "started_utc": started.isoformat(timespec="seconds"),
        "dataset_cases": len(cases),
        "dataset_digest": digest,
        "strata": _count([c.stratum for c in cases]),
        "prompts": prompts.registry(),
        # The bands actually in force, read from artifacts/thresholds.json --
        # not the cold-start defaults in config. Reporting the config values
        # here would put numbers in the header that the run never used.
        "bands": _active_bands(cfg),
        "repeat": run_cfg.repeat,
        "judged": run_cfg.judge,
        "python": platform.python_version(),
        **meta,
    }


def _active_bands(cfg: Settings) -> dict[str, Any]:
    path = cfg.artifact_dir / "thresholds.json"
    if not path.exists():
        return {"lo": cfg.auto_approve_below, "hi": cfg.auto_block_above, "source": "config"}
    report = json.loads(path.read_text(encoding="utf-8"))
    chosen = report["chosen"]
    return {
        "lo": chosen["lo"],
        "hi": chosen["hi"],
        "source": "artifacts/thresholds.json",
        "nfr_recall_met": report.get("nfr_recall_met"),
        "expected_review_rate": chosen.get("review_rate"),
    }


def _case_row(r: CaseResult) -> dict[str, Any]:
    return {
        "case_id": r.case_id,
        "stratum": r.stratum,
        "expected": r.expected_action,
        "recommendation": r.recommendation,
        "agrees": r.agrees,
        "confidence": r.confidence,
        "repairs": r.repairs,
        "policy_violations": r.policy_violations,
        "grounded": f"{r.grounded_claims}/{r.total_claims}",
        "narrative_score": (r.judge or {}).get("narrative_score"),
        "hallucinated": (r.judge or {}).get("hallucinated"),
        "latency_ms": r.latency_ms,
        "cost_usd": r.cost_usd,
    }


def _mean_of(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metrics averaged across repeats.

    Only the scalar top-level metrics are averaged; nested structures come from
    the first run, since averaging a histogram of recommendations across runs
    produces a shape no single run ever had.
    """
    out = dict(runs[0])
    for key in ("agreement_macro", "agreement_micro", "decisive_rate", "escalation_rate"):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return out


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
