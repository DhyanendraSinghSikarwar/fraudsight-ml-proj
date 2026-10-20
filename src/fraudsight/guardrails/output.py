"""Output policy: what makes a structurally valid brief still unsafe to act on.

Pydantic validates *shape*. These rules validate *meaning*, and they exist
because the interesting failures all produce well-formed JSON:

* a brief that says `block` with confidence 0.31 -- schema-valid, and a wrongly
  declined customer
* a brief citing `$4,182.00` when no tool ever returned that number -- schema
  valid, and a hallucination
* a brief listing `case_7734` in `similar_cases` when retrieval returned four
  other IDs -- schema-valid, and a fabricated citation
* a brief with eight risk-increasing signals and nothing mitigating -- schema
  valid, and not an investigation

Each rule returns `warn` or `block`. `block` sends the case to the repair loop
and, if repair fails, to a human -- it never silently rewrites the model's
output, because a policy layer that edits a brief makes the eval numbers
describe the policy layer rather than the model.

The groundedness rule is the load-bearing one and it is deliberately narrow: it
checks *numeric* claims only. Checking prose against evidence is what the LLM
judge does, with a measured kappa; a regex pretending to do it would produce a
confident number that means nothing. Numbers, though, are mechanical -- either
the tools returned 14 or they did not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fraudsight.triage.evidence import EvidenceBundle
from fraudsight.triage.schema import Direction, PolicyViolation, Recommendation, TriageBrief

# Below this, a recommendation to act unilaterally is not defensible; the case
# belongs with an analyst. Set from the cost matrix: at a $180 fraud loss and a
# $25 false decline, an autonomous block wants to be right about four times out
# of five before it beats escalating.
MIN_CONFIDENCE_TO_ACT = 0.80
MIN_CONFIDENCE_FOR_BLOCK = 0.85

# A number in prose that is not in the evidence. Matches currency, percentages,
# multipliers ("9.1x"), and bare integers. Two exclusions, both learned from
# false positives that would have made the metric meaningless:
#
#   ordinals    "95th percentile" is a label, not a claim of 95
#   identifiers "p95", "V258", "card1", "id_31" are field names, not claims
#
# A groundedness check that fires on those is measuring its own regex rather
# than the model's honesty.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\$?\d[\d,]*\.?\d*%?(?!(?:st|nd|rd|th)\b)")


@dataclass(frozen=True)
class PolicyReport:
    violations: list[PolicyViolation]

    @property
    def blocked(self) -> bool:
        return any(v.severity == "block" for v in self.violations)

    @property
    def clean(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "blocked": self.blocked,
            "violations": [v.model_dump() for v in self.violations],
        }

    def repair_instruction(self) -> str:
        lines = [f"- {v.rule}: {v.detail}" for v in self.violations if v.severity == "block"]
        return "The brief violated output policy:\n" + "\n".join(lines)


def validate(brief: TriageBrief, bundle: EvidenceBundle) -> PolicyReport:
    v: list[PolicyViolation] = []

    # -- confidence must match the strength of the action ----------------
    acts_unilaterally = brief.recommendation in (Recommendation.APPROVE, Recommendation.BLOCK)
    if acts_unilaterally and brief.confidence < MIN_CONFIDENCE_TO_ACT:
        v.append(
            PolicyViolation(
                rule="low_confidence_unilateral_action",
                detail=(
                    f"recommendation={brief.recommendation} at confidence "
                    f"{brief.confidence:.2f}; below {MIN_CONFIDENCE_TO_ACT} the case must be "
                    f"escalate_to_analyst"
                ),
                severity="block",
            )
        )
    if brief.recommendation is Recommendation.BLOCK and brief.confidence < MIN_CONFIDENCE_FOR_BLOCK:
        v.append(
            PolicyViolation(
                rule="block_confidence_floor",
                detail=(
                    f"blocking a customer at confidence {brief.confidence:.2f} is below the "
                    f"{MIN_CONFIDENCE_FOR_BLOCK} floor for an irreversible decline"
                ),
                severity="block",
            )
        )

    # -- an investigation considers both sides ---------------------------
    if brief.recommendation is Recommendation.BLOCK and not brief.has_mitigating_evidence():
        v.append(
            PolicyViolation(
                rule="one_sided_evidence",
                detail=(
                    "recommending block with no risk_decreasing evidence; state what points "
                    "the other way, or say explicitly that nothing does"
                ),
                severity="warn",
            )
        )

    # -- incomplete evidence must not be laundered into confidence -------
    if brief.missing_information and brief.confidence > 0.9 and acts_unilaterally:
        v.append(
            PolicyViolation(
                rule="confident_despite_gaps",
                detail=(
                    f"confidence {brief.confidence:.2f} with "
                    f"{len(brief.missing_information)} unresolved evidence gap(s)"
                ),
                severity="block",
            )
        )

    # -- citations must exist --------------------------------------------
    retrieved = {
        str(c.get("case_id"))
        for c in bundle.tool_fields("similar_cases").get("cases", [])
    }
    invented = [c for c in brief.similar_cases if c not in retrieved]
    if invented:
        v.append(
            PolicyViolation(
                rule="fabricated_case_citation",
                detail=(
                    f"cited case id(s) {invented} that retrieval never returned "
                    f"(available: {sorted(retrieved) or 'none'})"
                ),
                severity="block",
            )
        )

    # -- tool attribution must be truthful --------------------------------
    ran = {name for name, r in bundle.tools.items() if r.found} | {"transaction"}
    misattributed = sorted({e.source_tool for e in brief.evidence} - ran)
    if misattributed:
        v.append(
            PolicyViolation(
                rule="misattributed_evidence",
                detail=(
                    f"evidence attributed to {misattributed}, which returned nothing for this "
                    f"transaction"
                ),
                severity="block",
            )
        )

    # -- numeric groundedness ---------------------------------------------
    allowed = _floats(bundle.claim_values())
    for e in brief.evidence:
        for token, decimals in _numbers(e.detail):
            if not _supported(token, decimals, allowed):
                v.append(
                    PolicyViolation(
                        rule="ungrounded_number",
                        detail=(
                            f"evidence '{e.signal}' cites {token:g}, which does not appear in "
                            f"any tool output for this transaction"
                        ),
                        severity="block",
                    )
                )

    # -- disagreement must be stated, not implied --------------------------
    #
    # "Did the model flag this?" is judged against the portfolio base rate, not
    # against 0.5 -- see rules.model_says_risky. Every transaction in the triage
    # band has an elevated score by construction, so a 0.5 cut would classify
    # almost all of them as the model saying "safe" and fire this rule on
    # briefs that agree with the model perfectly well.
    from fraudsight.triage.rules import model_says_risky

    score = float(bundle.model.get("score") or 0.0)
    flagged = model_says_risky(dict(bundle.model))
    agent_says_risky = brief.recommendation in (
        Recommendation.BLOCK,
        Recommendation.REQUEST_STEP_UP,
    )
    if brief.agrees_with_model_score and flagged != agent_says_risky:
        v.append(
            PolicyViolation(
                rule="inconsistent_agreement_flag",
                detail=(
                    f"agrees_with_model_score=True but the model scored {score:.3f} "
                    f"({'flagged' if flagged else 'not flagged'} against the base rate) and "
                    f"the recommendation is {brief.recommendation}"
                ),
                severity="warn",
            )
        )

    return PolicyReport(violations=v)


def _numbers(text: str) -> set[tuple[float, int]]:
    """Numeric claims worth checking, each with the precision it was stated at.

    Carrying the decimal count matters. A brief that says "1.5% of its history"
    about a stored rate of 0.015385 is correct to the precision it chose to
    display, and a checker that demands 1.5385 flags honest rounding as
    hallucination. Comparing at the *claimed* precision is the only way the
    metric measures invention rather than formatting.

    Sub-10 integers are skipped: "3 of 5 cases" is ordinary English, and
    treating every small integer as a citable claim would swamp the metric with
    false positives.
    """
    out: set[tuple[float, int]] = set()
    for m in _NUMBER_RE.finditer(text):
        cleaned = m.group(0).strip("$%").replace(",", "").rstrip(".")
        if not cleaned:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if value < 10 and value == int(value):
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        out.add((value, decimals))
    return out


# Residual relative tolerance, for claims whose precision cannot be recovered
# (a bare "9x" against a stored 9.0004). Deliberately tight -- the precision
# rule above does the real work, and a wide tolerance here would let genuinely
# invented numbers through by landing near a real one.
_REL_TOLERANCE = 0.011


def _supported(value: float, decimals: int, allowed: set[float]) -> bool:
    if value in allowed:
        return True
    # Primary rule: does any evidence value round to exactly what was claimed?
    if any(round(a, decimals) == round(value, decimals) for a in allowed):
        return True
    scale = max(abs(value), 1.0)
    return any(abs(value - a) <= _REL_TOLERANCE * max(scale, abs(a)) for a in allowed)


def evidence_is_grounded(brief: TriageBrief, bundle: EvidenceBundle) -> tuple[int, int]:
    """(grounded_claims, total_claims) for the metrics table."""
    allowed = _floats(bundle.claim_values())
    total = grounded = 0
    for e in brief.evidence:
        for token, decimals in _numbers(e.detail):
            total += 1
            grounded += int(_supported(token, decimals, allowed))
    return grounded, total


def _floats(values: set[str]) -> set[float]:
    out: set[float] = set()
    for v in values:
        try:
            out.add(float(str(v).replace(",", "").strip("$%")))
        except (TypeError, ValueError):
            continue
    # Percentages in prose ("25.0% of its history") cite a rate stored as 0.25.
    out |= {v * 100.0 for v in list(out) if 0.0 < v <= 1.0}
    return out


def has_only_risk_increasing(brief: TriageBrief) -> bool:
    return all(e.direction is Direction.RISK_INCREASING for e in brief.evidence)
