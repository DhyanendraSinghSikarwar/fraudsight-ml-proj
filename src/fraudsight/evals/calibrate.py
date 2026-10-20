"""Judge calibration: measure the instrument before trusting its readings.

**An honest caveat, stated up front because it changes how the number should be
read.** The blueprint calls for Cohen's kappa against 30 *human* labels. The
labels shipped in `evals/datasets/reference_labels.jsonl` were not produced by a
human -- they were produced by the deterministic rubric in
`_reference_label()` below. Kappa against them therefore measures
**judge-versus-rules agreement**, not judge-versus-human agreement, and it is
reported under that name everywhere it appears.

That is a real gap and it is recorded in docs/scope.md rather than papered over.
What closes it is a person spending about ninety minutes with
`evals/labelling_protocol.md` and `fraudsight label`, which writes
`human_labels.jsonl`; this module prefers that file whenever it exists and
relabels every output `judge_vs_human`. The plumbing is complete. The labour is
not, and pretending otherwise would be exactly the kind of unearned number this
whole eval section exists to avoid.

Why ship the rules-based reference at all: it still catches a judge that has
broken outright. A judge that marks a brief citing a number found nowhere in the
evidence as "supported" disagrees with the rule check, kappa collapses, and the
gate fails. It cannot catch a judge with subtly bad taste about semantic
groundedness. Those are different claims and the report makes the distinction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fraudsight.config import Settings, settings as default_settings
from fraudsight.evals.judges import (
    CalibrationReport,
    Judge,
    calibrate,
    narrative_penalties,
)
from fraudsight.guardrails.output import evidence_is_grounded
from fraudsight.triage.evidence import EvidenceBundle
from fraudsight.triage.schema import TriageBrief

HUMAN_LABELS = Path("evals/datasets/human_labels.jsonl")
REFERENCE_LABELS = Path("evals/datasets/reference_labels.jsonl")

CALIBRATION_SET_SIZE = 30


def label_source(cfg: Settings | None = None) -> tuple[Path, str]:
    """Human labels win whenever they exist. The returned tag propagates into
    every metric name so a reader always knows which comparison they are
    looking at."""
    cfg = cfg or default_settings
    root = cfg.artifact_dir.parent
    human = root / HUMAN_LABELS
    if human.exists():
        return human, "judge_vs_human"
    return root / REFERENCE_LABELS, "judge_vs_rules"


def _reference_label(brief: TriageBrief, bundle: EvidenceBundle) -> dict[str, Any]:
    """The deterministic rubric. Mechanical by design -- it is a floor, not a
    stand-in for judgement."""
    grounded, total = evidence_is_grounded(brief, bundle)
    numeric_clean = total == 0 or grounded == total

    retrieved = {
        str(c.get("case_id")) for c in bundle.tool_fields("similar_cases").get("cases", [])
    }
    citations_clean = all(c in retrieved for c in brief.similar_cases)

    ran = {n for n, r in bundle.tools.items() if r.found} | {"transaction"}
    attribution_clean = not ({e.source_tool for e in brief.evidence} - ran)

    grounded_overall = numeric_clean and citations_clean and attribution_clean

    # Same three rubric penalties the judge applies, from the same function.
    # A second hand-rolled copy is how the two drifted apart the first time.
    penalties = narrative_penalties(brief.model_dump(mode="json"))

    return {
        "transaction_id": brief.transaction_id,
        "grounded": int(grounded_overall),
        "narrative_score": max(1, min(5, 5 - len(penalties))),
        "labeller": "deterministic-rubric",
        "rubric_version": "v1",
    }


def build_reference_labels(
    pairs: list[tuple[TriageBrief, EvidenceBundle]],
    *,
    path: Path | None = None,
    limit: int = CALIBRATION_SET_SIZE,
) -> Path:
    cfg = default_settings
    p = path or (cfg.artifact_dir.parent / REFERENCE_LABELS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "__header__": (
                        "Rules-based reference labels, NOT human labels. Kappa computed "
                        "against these measures judge-versus-rules agreement. See "
                        "evals/labelling_protocol.md to produce human_labels.jsonl, which "
                        "takes precedence when present."
                    ),
                    "rubric_version": "v1",
                }
            )
            + "\n"
        )
        for brief, bundle in pairs[:limit]:
            fh.write(json.dumps(_reference_label(brief, bundle)) + "\n")
    return p


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `fraudsight calibrate --build-reference` or produce "
            f"human labels per evals/labelling_protocol.md"
        )
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "__header__" in row:
            continue
        out[str(row["transaction_id"])] = row
    return out


async def calibrate_judge(
    judge: Judge,
    pairs: list[tuple[TriageBrief, EvidenceBundle]],
    cfg: Settings | None = None,
) -> dict[str, Any]:
    """Run the judge over the labelled set and report kappa for both metrics."""
    cfg = cfg or default_settings
    path, tag = label_source(cfg)
    labels = load_labels(path)

    scored = [(b, e) for b, e in pairs if b.transaction_id in labels]
    if not scored:
        raise ValueError(
            f"none of the {len(pairs)} briefs match the {len(labels)} labelled cases in "
            f"{path.name}; rebuild the labels against the current eval set"
        )

    rows = await judge.judge_many(scored)

    judge_grounded = [int(not r["hallucinated"]) for r in rows]
    human_grounded = [int(labels[r["transaction_id"]]["grounded"]) for r in rows]
    judge_narrative = [int(r["narrative_score"]) for r in rows]
    human_narrative = [int(labels[r["transaction_id"]]["narrative_score"]) for r in rows]

    reports: list[CalibrationReport] = [
        calibrate(
            judge_grounded,
            human_grounded,
            metric=f"groundedness[{tag}]",
            threshold=cfg.gate_min_judge_kappa,
        ),
        calibrate(
            judge_narrative,
            human_narrative,
            metric=f"narrative_quality[{tag}]",
            threshold=cfg.gate_min_judge_kappa,
            weights="quadratic",
        ),
    ]

    independent = tag == "judge_vs_human"
    return {
        "comparison": tag,
        # The field the gate keys on. When both sides of the comparison are
        # rule-based they share `narrative_penalties`, so a high kappa is a
        # tautology rather than evidence -- it says the code agrees with itself.
        # Reporting it as calibration would be the single most misleading number
        # in this repository, so it is labelled and the gate refuses to rely on
        # any judge-derived metric until a human has labelled the set.
        "independent": independent,
        "labels_from": str(path.name),
        "labeller": next(iter(labels.values()), {}).get("labeller", "unknown"),
        "n": len(rows),
        "caveat": (
            "NOT CALIBRATION. Both the judge and the reference labels come from the same "
            "rule set, so kappa here only detects a judge that has broken outright -- a "
            "high value is self-agreement, not accuracy. Judge-derived metrics "
            "(hallucination rate, narrative score) are therefore UNVALIDATED. See "
            "evals/labelling_protocol.md."
            if not independent
            else "Kappa against human labels produced per evals/labelling_protocol.md."
        ),
        "reports": [r.as_dict() for r in reports],
        "usable": independent and all(r.usable for r in reports),
    }


def save(report: dict[str, Any], cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    p = cfg.artifact_dir / "judge_calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p
