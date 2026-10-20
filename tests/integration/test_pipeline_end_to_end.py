"""End-to-end behaviour of the assembled system.

Marked `integration` because they need the trained artifacts and the corpus.
They still run with no vendor account -- the deterministic provider is a real
implementation of the protocol, which is what keeps this suite runnable in CI.
"""

from __future__ import annotations

import pytest

from fraudsight.scoring.serve import Band
from fraudsight.triage.state import Outcome
from tests.conftest import needs_artifacts, needs_corpus, needs_evalset

pytestmark = [pytest.mark.integration, needs_artifacts, needs_corpus, needs_evalset]


async def test_triage_produces_a_valid_brief(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    decision = await pipeline.process(eval_cases[0].transaction)
    assert decision.band is Band.TRIAGE
    state = decision.state
    assert state is not None
    assert state.outcome is Outcome.COMPLETED
    brief = state.brief
    assert brief is not None
    assert 2 <= len(brief.evidence) <= 8
    assert 80 <= len(brief.narrative) <= 900
    assert 0.0 <= brief.confidence <= 1.0


async def test_every_node_runs_in_order(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    """Redaction is first and unconditional. A reordering that let a tool call
    precede it would put raw PII in the tool results and the traces."""
    decision = await pipeline.process(eval_cases[0].transaction)
    names = [s.name for s in decision.state.spans]
    assert names[0] == "guardrails.redact_pii"
    for expected in (
        "tools.gather",
        "retrieval.similar_cases",
        "llm.produce_brief",
        "guardrails.validate_output",
    ):
        assert expected in names, f"{expected} never ran"
    assert names.index("tools.gather") < names.index("retrieval.similar_cases")


async def test_evidence_is_traceable_to_tools_that_ran(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    decision = await pipeline.process(eval_cases[0].transaction)
    state = decision.state
    ran = {n for n, r in state.bundle.tools.items() if r.found} | {"transaction"}
    assert {e.source_tool for e in state.brief.evidence} <= ran


async def test_cited_cases_were_actually_retrieved(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    decision = await pipeline.process(eval_cases[0].transaction)
    state = decision.state
    retrieved = {c["case_id"] for c in state.bundle.similar_cases}
    assert set(state.brief.similar_cases) <= retrieved


async def test_auto_approve_band_never_calls_the_agent(pipeline, scorer) -> None:  # type: ignore[no-untyped-def]
    """The whole thesis: ~87% of volume must not touch an LLM. If a cheap-band
    transaction produces a triage state, the routing is broken and every cost
    number in the README is fiction.

    Rows come from the val split rather than the eval set, because the eval set
    is sampled from the triage band by construction and contains no
    auto-approve case to find.
    """
    from fraudsight.scoring.dataset import load_split

    split = load_split("val", scorer.stats)
    probs = scorer.score_frame(split.X.head(2_000))
    low = [i for i, p in enumerate(probs) if p < scorer.lo][:5]
    assert low, "no auto-approve rows in the first 2,000 val transactions"

    for i in low:
        row = split.X.iloc[i].to_dict()
        row["TransactionID"] = int(split.txn_id[i])
        row["TransactionDT"] = int(split.time[i])
        decision = await pipeline.process(row)
        assert decision.band is Band.AUTO_APPROVE
        assert decision.state is None, "an auto-approved transaction reached the agent"
        assert decision.cost_usd == 0.0
        assert decision.score_ms < 50.0


async def test_provider_swap_is_a_config_change(eval_cases) -> None:  # type: ignore[no-untyped-def]
    """The claim in the README, tested. Nothing but config changes, and the
    pipeline still produces a valid brief."""
    from fraudsight.config import settings
    from fraudsight.triage.pipeline import TriagePipeline

    for provider in ("deterministic",):
        cfg = settings.model_copy(
            update={"primary_provider": provider, "fallback_provider": None}
        )
        pipe = TriagePipeline.build(cfg, write_traces=False)
        decision = await pipe.process(eval_cases[0].transaction)
        assert decision.state.brief is not None
        assert decision.state.usage.provider == provider


async def test_degrades_to_a_brief_when_the_primary_provider_fails(eval_cases) -> None:  # type: ignore[no-untyped-def]
    """Graceful degradation: provider down must not drop the transaction."""
    from fraudsight.config import settings
    from fraudsight.llm import build_ledger
    from fraudsight.llm.base import ProviderUnavailable
    from fraudsight.llm.deterministic import DeterministicClient
    from fraudsight.llm.router import LLMRouter
    from fraudsight.triage.pipeline import TriagePipeline

    class DeadClient(DeterministicClient):
        provider_name = "anthropic"

        async def structured(self, **kw):  # type: ignore[no-untyped-def]
            raise ProviderUnavailable("simulated outage", provider="anthropic")

    router = LLMRouter(
        DeadClient(), [DeterministicClient()], ledger=build_ledger(settings), max_attempts=2
    )
    pipe = TriagePipeline.build(settings, router=router, write_traces=False)
    decision = await pipe.process(eval_cases[0].transaction)

    assert decision.state.brief is not None, "a provider outage dropped the transaction"
    assert decision.state.degraded is True
    assert decision.state.usage.provider == "deterministic"


async def test_traces_are_written_and_parseable(eval_cases, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from fraudsight.config import settings
    from fraudsight.triage.pipeline import TriagePipeline

    cfg = settings.model_copy(
        update={
            "primary_provider": "deterministic",
            "fallback_provider": None,
            "trace_dir": tmp_path,
        }
    )
    pipe = TriagePipeline.build(cfg, write_traces=True)
    await pipe.process(eval_cases[0].transaction)

    lines = (tmp_path / "triage.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[-1])
    for key in ("transaction_id", "outcome", "score", "cost_usd", "spans", "prompt_ids"):
        assert key in row, f"trace is missing {key}"


async def test_summary_reports_the_funnel(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    for case in eval_cases[:10]:
        await pipeline.process(case.transaction)
    summary = pipeline.summary()
    assert summary["transactions"] == 10
    assert summary["scoring_latency"]["p99_ms"] < 50.0
    assert sum(summary["band_counts"].values()) == 10
