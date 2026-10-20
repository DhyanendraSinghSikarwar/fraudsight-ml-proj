"""Shared fixtures.

Tests that need trained artifacts are skipped rather than failed when the
artifacts are absent, so `pytest` on a fresh clone reports "12 passed, 9
skipped" instead of a wall of red that tells a new contributor nothing. CI runs
`make artifacts` first, so nothing is silently skipped where it matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fraudsight.config import settings

ARTIFACTS_REQUIRED = [
    settings.artifact_dir / "model.txt",
    settings.artifact_dir / "calibrator.json",
    settings.artifact_dir / "entity_stats.json",
]

needs_artifacts = pytest.mark.skipif(
    not all(p.exists() for p in ARTIFACTS_REQUIRED),
    reason="trained artifacts missing -- run `make artifacts`",
)

needs_corpus = pytest.mark.skipif(
    not (settings.artifact_dir / "cases" / "cases.jsonl").exists(),
    reason="case corpus missing -- run `fraudsight cases`",
)

needs_evalset = pytest.mark.skipif(
    not (Path(settings.artifact_dir.parent) / "evals" / "datasets" / "triage_holdout.jsonl").exists(),
    reason="eval set missing -- run `fraudsight evalset`",
)


@pytest.fixture(scope="session")
def profiles() -> Any:
    from fraudsight.data.profiles import ProfileTables

    return ProfileTables.load(settings.artifact_dir / "profiles")


@pytest.fixture(scope="session")
def scorer() -> Any:
    from fraudsight.scoring.serve import Scorer

    return Scorer.load()


@pytest.fixture(scope="session")
def eval_cases() -> Any:
    from fraudsight.data.cases import load_eval_cases

    return load_eval_cases()


@pytest.fixture
def sample_transaction(eval_cases: Any) -> dict[str, Any]:
    return dict(eval_cases[0].transaction)


@pytest.fixture
async def pipeline() -> Any:
    """A pipeline pinned to the deterministic provider.

    Unit and security tests must never depend on a vendor account or on network
    behaviour; a test suite that only passes with an API key present is a test
    suite that stops running.
    """
    from fraudsight.triage.pipeline import TriagePipeline

    cfg = settings.model_copy(
        update={"primary_provider": "deterministic", "fallback_provider": None}
    )
    return TriagePipeline.build(cfg, write_traces=False)
