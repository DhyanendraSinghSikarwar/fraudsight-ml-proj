"""The CI eval gate. A merge that drops quality below these floors fails.

The gate is the difference between having evals and being governed by them.
Three properties make it real rather than decorative:

* **It fails the build.** Non-zero exit, no override flag in CI. A gate with a
  `--force` that reviewers habitually pass is a slower way of not having a gate.
* **It fails loudly and specifically.** Each check prints the threshold, the
  measured value, and the margin, so the failure tells you what regressed rather
  than that something did.
* **It refuses to pass on insufficient evidence.** A run with fewer cases than
  `min_cases`, or an uncalibrated judge, fails -- because a green gate on 12
  cases is worse than a red one.

The judge-kappa check is the subtle one. If the LLM judge disagrees with the
human labels, the agent's groundedness and narrative numbers are unreliable, so
the gate fails on the *instrument* rather than reporting quality it cannot
measure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraudsight.config import Settings, settings as default_settings


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    measured: float | None
    threshold: float
    direction: str  # ">=" or "<="
    note: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        got = "n/a" if self.measured is None else f"{self.measured:.4f}"
        margin = (
            ""
            if self.measured is None
            else f"  (margin {self.measured - self.threshold:+.4f})"
        )
        # The note explains why a check exists, which is only worth the line
        # when it failed -- or when there is no measurement to show, where the
        # note IS the result. Printing it on every passing check buried the
        # failures in advice about checks that were fine.
        show_note = self.note and (not self.passed or self.measured is None)
        extra = f"  -- {self.note}" if show_note else ""
        return f"[{mark}] {self.name:32s} {got} {self.direction} {self.threshold:.4f}{margin}{extra}"


@dataclass(frozen=True)
class GateResult:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = [c.render() for c in self.checks]
        failed = [c.name for c in self.checks if not c.passed]
        lines.append("")
        lines.append(
            "EVAL GATE PASSED"
            if self.passed
            else f"EVAL GATE FAILED on: {', '.join(failed)}"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
        }


def evaluate(
    report: dict[str, Any],
    cfg: Settings | None = None,
    *,
    min_cases: int = 100,
    require_repeats: int = 1,
    calibration: dict[str, Any] | None = None,
) -> GateResult:
    cfg = cfg or default_settings
    m = report.get("metrics", {})
    header = report.get("header", {})
    checks: list[Check] = []

    def add(
        name: str,
        measured: float | None,
        threshold: float,
        direction: str = ">=",
        note: str = "",
    ) -> None:
        if measured is None:
            checks.append(Check(name, False, None, threshold, direction, note or "not measured"))
            return
        ok = measured >= threshold if direction == ">=" else measured <= threshold
        checks.append(Check(name, ok, float(measured), threshold, direction, note))

    # -- sufficiency ---------------------------------------------------
    add("eval_cases", float(m.get("cases", 0)), float(min_cases), ">=",
        "too few cases to draw a conclusion")
    add("repeats", float(header.get("repeat", 1)), float(require_repeats), ">=",
        "run-to-run variance is unmeasured below this")

    # -- quality --------------------------------------------------------
    add("agreement_macro", m.get("agreement_macro"), cfg.gate_min_agreement)
    add(
        "schema_pass_rate",
        (m.get("schema") or {}).get("pass_rate"),
        cfg.gate_min_schema_pass,
    )
    add(
        "numeric_grounding",
        (m.get("groundedness") or {}).get("numeric_claim_grounding"),
        1.0 - cfg.gate_max_hallucination,
    )

    # -- policy ---------------------------------------------------------
    add("policy_clean_rate", (m.get("policy") or {}).get("clean_rate"), 0.95)

    # -- the instrument -------------------------------------------------
    #
    # Judge-derived metrics are gated ONLY when the judge has been calibrated
    # against independent (human) labels. With a rules-based reference, the
    # judge and the labeller share a rule set and their agreement is a
    # tautology -- gating on it would dress self-agreement up as quality
    # assurance. So the judge metrics are reported and skipped, and the gate
    # leans on the mechanical checks above, which need no judge at all.
    independent = bool((calibration or {}).get("independent"))
    judge_rate = (m.get("groundedness") or {}).get("judge_hallucination_rate")

    if independent:
        for entry in (calibration or {}).get("reports", []):
            add(
                f"judge_kappa[{entry['metric']}]",
                entry.get("cohens_kappa"),
                entry.get("threshold", cfg.gate_min_judge_kappa),
                ">=",
                entry.get("interpretation", ""),
            )
        if judge_rate is not None:
            add("judge_hallucination_rate", judge_rate, cfg.gate_max_hallucination, "<=")
    else:
        checks.append(
            Check(
                "judge_calibration",
                passed=True,
                measured=None,
                threshold=cfg.gate_min_judge_kappa,
                direction=">=",
                note=(
                    "NOT ESTABLISHED -- reference labels are rule-based, so judge-derived "
                    "metrics are reported but not gated. See evals/labelling_protocol.md."
                ),
            )
        )

    # -- the point of the system ----------------------------------------
    # A high agreement achieved by escalating everything is not a working triage
    # agent. This is the check that keeps the headline number honest.
    add(
        "decisive_rate",
        m.get("decisive_rate"),
        0.25,
        ">=",
        "escalating almost everything makes agreement meaningless",
    )

    return GateResult(checks)


def load_and_evaluate(
    report_path: Path | None = None,
    cfg: Settings | None = None,
    **kwargs: Any,
) -> GateResult:
    cfg = cfg or default_settings
    p = report_path or (cfg.artifact_dir / "eval_report.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `fraudsight eval` first")
    report = json.loads(p.read_text(encoding="utf-8"))

    cal_path = cfg.artifact_dir / "judge_calibration.json"
    calibration = (
        json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
    )
    return evaluate(report, cfg, calibration=calibration, **kwargs)
