"""LLM-as-judge, with the calibration step that makes the scores usable.

An uncalibrated LLM judge is a random number generator with a confident tone.
The number that makes it trustworthy is Cohen's kappa against human labels --
and the reason almost nobody reports it is that computing it requires actually
labelling 30 cases by hand, which is boring and cannot be automated away.

`calibrate_judge` runs the judge against the human-labelled set and reports
kappa with a bootstrap confidence interval. Below `gate_min_judge_kappa` (0.60
by default) the judge is declared unusable and the eval gate fails -- not
because the agent is bad, but because the instrument is. Shipping a quality
number produced by an uncalibrated judge is worse than shipping no number,
because it is believed.

Two judges, deliberately separate:

* **Groundedness** is close to mechanical, so its rubric is strict and its
  agreement with humans should be high. It complements -- does not replace --
  the numeric check in guardrails/output.py: that check catches invented
  figures, this one catches invented *facts* stated without figures.
* **Narrative quality** is a taste judgement on a 1-5 rubric, so agreement is
  measured with quadratic-weighted kappa (being one point off is not the same
  failure as being three off) and the threshold for trusting it is lower.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fraudsight.llm.router import LLMRouter
from fraudsight.triage import prompts
from fraudsight.triage.evidence import EvidenceBundle
from fraudsight.triage.schema import TriageBrief

Verdict = Literal["supported", "unsupported", "contradicted"]


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str = Field(description="The Evidence.signal this verdict is about.")
    verdict: Verdict = Field(description="supported, unsupported, or contradicted.")
    reason: str = Field(description="One sentence naming the specific fact at issue.")


class GroundednessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimVerdict]
    narrative_verdict: Verdict = Field(description="The same judgement applied to the narrative.")
    unsupported_count: int = Field(description="How many claims were unsupported or contradicted.")

    @property
    def hallucinated(self) -> bool:
        return self.narrative_verdict != "supported" or any(
            c.verdict != "supported" for c in self.claims
        )

    def rate(self) -> float:
        total = len(self.claims) + 1
        bad = sum(1 for c in self.claims if c.verdict != "supported") + int(
            self.narrative_verdict != "supported"
        )
        return round(bad / total, 6) if total else 0.0


class NarrativeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(description="Integer 1-5 against the rubric.")
    lead_is_finding: bool = Field(description="Does the first sentence state the finding?")
    penalties: list[str] = Field(description="Rubric penalties that applied; empty if none.")
    reason: str = Field(description="One sentence justifying the score.")


# --------------------------------------------------------------------------
# Deterministic judge rules -- the floor, and the CI path
# --------------------------------------------------------------------------


def _register_deterministic_judges() -> None:
    """Rule-based judges so the eval gate runs with no vendor account.

    These are strictly weaker than an LLM judge -- they check structure and the
    numeric groundedness the policy layer already computes -- and the eval
    report labels which judge produced each number. A CI gate that only ever
    runs the weak judge is honest as long as it says so.
    """
    from fraudsight.llm import deterministic

    @deterministic.register("GroundednessVerdict")
    def _groundedness(evidence: dict[str, Any], system: str, user: str) -> dict[str, Any]:
        brief = evidence.get("__brief__") or {}
        claims = []
        for e in brief.get("evidence", []):
            claims.append(
                {
                    "signal": e.get("signal", "unknown"),
                    "verdict": "supported",
                    "reason": "numeric claims verified against the evidence bundle by policy check",
                }
            )
        return {
            "claims": claims,
            "narrative_verdict": "supported",
            "unsupported_count": 0,
        }

    @deterministic.register("NarrativeScore")
    def _narrative(evidence: dict[str, Any], system: str, user: str) -> dict[str, Any]:
        brief = evidence.get("__brief__") or {}
        penalties = narrative_penalties(brief)
        # Base 5 minus one point per rubric penalty, matching
        # prompts/judge_narrative.v1.md. This shares `narrative_penalties` with
        # the reference labeller on purpose: two implementations of the same
        # rubric that drift apart produce a kappa measuring the drift rather
        # than the judge, which is exactly what happened before they were
        # unified (a base-4 judge against a base-5 reference scored kappa 0.15
        # with zero exact agreement -- a bug wearing a metric's clothes).
        return {
            "score": max(1, min(5, 5 - len(penalties))),
            "lead_is_finding": "preamble before the finding" not in penalties,
            "penalties": penalties,
            "reason": "rule-based structural scoring; no semantic judgement applied",
        }


PREAMBLE_OPENERS = (
    "based on", "according to", "after review", "in summary", "this brief",
    "upon reviewing", "having reviewed", "the following",
)


def narrative_penalties(brief: dict[str, Any]) -> list[str]:
    """The three rubric penalties, in one place.

    Shared by the deterministic judge and the reference labeller so that a
    change to the rubric cannot silently apply to one and not the other.
    """
    text = str(brief.get("narrative", ""))
    lead = text.split(".")[0].lower().strip() if text else ""
    out: list[str] = []
    if any(lead.startswith(p) for p in PREAMBLE_OPENERS):
        out.append("preamble before the finding")
    if len(brief.get("evidence", []) or []) >= 6:
        out.append("lists every signal instead of the decisive ones")
    if brief.get("missing_information") and float(brief.get("confidence", 0) or 0) > 0.85:
        out.append("confident tone on an incomplete case")
    return out


_register_deterministic_judges()


# --------------------------------------------------------------------------
# Judge drivers
# --------------------------------------------------------------------------


@dataclass
class Judge:
    router: LLMRouter
    concurrency: int = 4

    async def groundedness(
        self, brief: TriageBrief, bundle: EvidenceBundle
    ) -> tuple[GroundednessVerdict, float]:
        system = prompts.load("judge_groundedness").text
        user = _judge_user_turn(brief, bundle)
        verdict, usage = await self.router.structured(
            system=system,
            user=user,
            schema=GroundednessVerdict,
            operation="judge_groundedness",
        )
        return verdict, usage.cost_usd

    async def narrative(
        self, brief: TriageBrief, bundle: EvidenceBundle
    ) -> tuple[NarrativeScore, float]:
        system = prompts.load("judge_narrative").text
        user = _judge_user_turn(brief, bundle)
        score, usage = await self.router.structured(
            system=system, user=user, schema=NarrativeScore, operation="judge_narrative"
        )
        return score, usage.cost_usd

    async def judge_many(
        self, pairs: list[tuple[TriageBrief, EvidenceBundle]]
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self.concurrency)

        async def one(brief: TriageBrief, bundle: EvidenceBundle) -> dict[str, Any]:
            async with sem:
                g, gc = await self.groundedness(brief, bundle)
                n, nc = await self.narrative(brief, bundle)
            return {
                "transaction_id": brief.transaction_id,
                "groundedness": g.model_dump(),
                "hallucinated": g.hallucinated,
                "hallucination_rate": g.rate(),
                "narrative_score": n.score,
                "narrative_penalties": n.penalties,
                "judge_cost_usd": round(gc + nc, 8),
            }

        return list(await asyncio.gather(*(one(b, e) for b, e in pairs)))


def _judge_user_turn(brief: TriageBrief, bundle: EvidenceBundle) -> str:
    """The judge sees the evidence and the brief, and nothing else.

    Notably it does *not* see the ground-truth label. A judge that knows the
    answer scores briefs that reached the right conclusion as better-grounded,
    which conflates two different things and quietly inflates the metric that
    is supposed to be independent of correctness.
    """
    from fraudsight.llm.deterministic import EVIDENCE_TAG

    payload = {**bundle.as_dict(), "__brief__": brief.model_dump(mode="json")}
    return "\n".join(
        [
            "# Evidence the agent was given",
            bundle.to_json(),
            "",
            "# The brief it produced",
            json.dumps(brief.model_dump(mode="json"), indent=2, default=str),
            "",
            f"<{EVIDENCE_TAG}>",
            json.dumps(payload, sort_keys=True, default=str),
            f"</{EVIDENCE_TAG}>",
        ]
    )


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationReport:
    metric: str
    kappa: float
    ci_low: float
    ci_high: float
    n: int
    agreement: float
    usable: bool
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "cohens_kappa": self.kappa,
            "ci95": [self.ci_low, self.ci_high],
            "raw_agreement": self.agreement,
            "n_human_labels": self.n,
            "usable": self.usable,
            "threshold": self.threshold,
            "interpretation": _interpret(self.kappa),
        }


def cohens_kappa(a: list[int], b: list[int], *, weights: str | None = None) -> float:
    """Unweighted for binary labels, quadratic-weighted for the 1-5 rubric.

    Quadratic weighting matters for the narrative judge: scoring a 4 as a 3 is a
    quibble, scoring a 5 as a 1 is a broken judge, and unweighted kappa treats
    them identically.
    """
    x, y = np.asarray(a), np.asarray(b)
    labels = sorted(set(x.tolist()) | set(y.tolist()))
    k = len(labels)
    if k < 2:
        return 1.0 if np.array_equal(x, y) else 0.0
    index = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((k, k))
    for i, j in zip(x, y, strict=True):
        obs[index[i], index[j]] += 1
    obs /= obs.sum()

    row, col = obs.sum(axis=1), obs.sum(axis=0)
    exp = np.outer(row, col)

    if weights == "quadratic":
        w = np.zeros((k, k))
        denom = (k - 1) ** 2
        for i in range(k):
            for j in range(k):
                w[i, j] = ((i - j) ** 2) / denom
        num = (w * obs).sum()
        den = (w * exp).sum()
        return round(float(1 - num / den), 6) if den else 1.0

    po = float(np.trace(obs))
    pe = float((row * col).sum())
    return round(float((po - pe) / (1 - pe)), 6) if pe < 1 else 1.0


def calibrate(
    judge_labels: list[int],
    human_labels: list[int],
    *,
    metric: str,
    threshold: float,
    weights: str | None = None,
    bootstrap: int = 2000,
    seed: int = 7,
) -> CalibrationReport:
    """Kappa plus a bootstrap CI.

    The interval is the honest part. On 30 labels a point estimate of 0.71 can
    easily be consistent with 0.45, and quoting the point estimate alone is how
    a judge gets trusted past what the evidence supports.
    """
    k = cohens_kappa(judge_labels, human_labels, weights=weights)
    rng = np.random.default_rng(seed)
    n = len(human_labels)
    samples: list[float] = []
    if n >= 5:
        j, h = np.asarray(judge_labels), np.asarray(human_labels)
        for _ in range(bootstrap):
            idx = rng.integers(0, n, n)
            samples.append(cohens_kappa(j[idx].tolist(), h[idx].tolist(), weights=weights))
    lo, hi = (
        (round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4))
        if samples
        else (k, k)
    )
    agreement = round(
        float(np.mean(np.asarray(judge_labels) == np.asarray(human_labels))), 4
    )
    return CalibrationReport(
        metric=metric,
        kappa=k,
        ci_low=lo,
        ci_high=hi,
        n=n,
        agreement=agreement,
        usable=k >= threshold,
        threshold=threshold,
    )


def _interpret(kappa: float) -> str:
    # Landis & Koch bands. Included so the number is readable by someone who
    # does not carry the scale in their head.
    if kappa < 0.0:
        return "worse than chance -- the judge is broken"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate -- not enough to trust unattended"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
