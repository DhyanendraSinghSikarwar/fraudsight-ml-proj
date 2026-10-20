"""Output policy: the rules that catch schema-valid but unsafe briefs."""

from __future__ import annotations

import pytest

from fraudsight.guardrails.output import evidence_is_grounded, validate
from fraudsight.tools.base import ToolResult
from fraudsight.triage.evidence import EvidenceBundle
from fraudsight.triage.schema import Direction, Evidence, Recommendation, TriageBrief


def _bundle(**overrides) -> EvidenceBundle:  # type: ignore[no-untyped-def]
    tools = {
        "customer_history": ToolResult(
            tool="customer_history",
            found=True,
            fields={"transactions": 154, "amount_p95": 240.0, "prior_confirmed_fraud": 0},
        ),
        "similar_cases": ToolResult(
            tool="similar_cases",
            found=True,
            fields={"cases": [{"case_id": "case_1"}, {"case_id": "case_2"}]},
        ),
    }
    bundle = EvidenceBundle(
        transaction_id="t1",
        transaction={"amount": 49.0},
        model={"score": 0.4, "band": "triage"},
        tools=tools,
        portfolio={"portfolio_fraud_rate": 0.034},
    )
    for k, v in overrides.items():
        setattr(bundle, k, v)
    return bundle


def _brief(**overrides) -> TriageBrief:  # type: ignore[no-untyped-def]
    payload = {
        "transaction_id": "t1",
        "narrative": (
            "The card is long established with 154 transactions and the amount is well "
            "within its usual range. Nothing in the retrieved evidence points to "
            "compromise. Recommending approval."
        ),
        "evidence": [
            Evidence(
                signal="established_customer",
                detail="Card has 154 transactions in its history.",
                direction=Direction.RISK_DECREASING,
                source_tool="customer_history",
                weight=0.5,
            ),
            Evidence(
                signal="model_risk_score",
                detail="The scorer put this transaction in the triage band at 0.40.",
                direction=Direction.RISK_INCREASING,
                source_tool="transaction",
                weight=0.4,
            ),
        ],
        "similar_cases": ["case_1"],
        "recommendation": Recommendation.APPROVE,
        "confidence": 0.86,
        "missing_information": [],
        "agrees_with_model_score": False,
    }
    payload.update(overrides)
    return TriageBrief(**payload)


def test_a_well_formed_brief_passes() -> None:
    assert validate(_brief(), _bundle()).clean


def test_low_confidence_unilateral_action_is_blocked() -> None:
    report = validate(_brief(confidence=0.55), _bundle())
    assert report.blocked
    assert any(v.rule == "low_confidence_unilateral_action" for v in report.violations)


def test_block_has_a_higher_confidence_floor_than_approve() -> None:
    """Declining a genuine customer is the irreversible one."""
    at_82 = validate(
        _brief(recommendation=Recommendation.BLOCK, confidence=0.82), _bundle()
    )
    assert any(v.rule == "block_confidence_floor" for v in at_82.violations)


def test_escalation_is_never_blocked_by_confidence() -> None:
    """Escalating on low confidence is the correct behaviour, not a violation."""
    report = validate(
        _brief(recommendation=Recommendation.ESCALATE, confidence=0.3), _bundle()
    )
    assert not report.blocked


def test_fabricated_case_citation_is_blocked() -> None:
    report = validate(_brief(similar_cases=["case_9999"]), _bundle())
    assert report.blocked
    assert any(v.rule == "fabricated_case_citation" for v in report.violations)


def test_evidence_attributed_to_a_tool_that_returned_nothing_is_blocked() -> None:
    ev = _brief().evidence
    ev[0] = ev[0].model_copy(update={"source_tool": "device_intel"})
    report = validate(_brief(evidence=ev), _bundle())
    assert report.blocked
    assert any(v.rule == "misattributed_evidence" for v in report.violations)


def test_ungrounded_number_is_blocked() -> None:
    ev = _brief().evidence
    ev[0] = ev[0].model_copy(
        update={"detail": "Card has 8,421 transactions in its history."}
    )
    report = validate(_brief(evidence=ev), _bundle())
    assert report.blocked
    assert any(v.rule == "ungrounded_number" for v in report.violations)


def test_rounding_to_the_stated_precision_is_grounded() -> None:
    """1.5385% displayed as '1.5%' is correct to the precision it chose. A
    checker that calls that a hallucination is measuring formatting."""
    bundle = _bundle()
    bundle.tools["customer_history"] = ToolResult(
        tool="customer_history",
        found=True,
        fields={"transactions": 65, "prior_fraud_rate": 0.015385, "amount_p95": 240.0},
    )
    ev = _brief().evidence
    ev[0] = ev[0].model_copy(
        update={"detail": "This card's history is 1.5% previously confirmed fraud."}
    )
    report = validate(_brief(evidence=ev), bundle)
    assert not report.blocked, [v.detail for v in report.violations]


def test_derived_ratios_are_grounded_once_recorded() -> None:
    """A ratio the evidence layer computed is part of the evidence. Exempting
    derived numbers from the check instead would open the hole the check exists
    to close."""
    bundle = _bundle()
    ev = _brief().evidence
    ev[0] = ev[0].model_copy(update={"detail": "The amount is 3.6x the card's p95."})

    assert validate(_brief(evidence=ev), bundle).blocked
    bundle.note_derived("amount_vs_customer_p95", 3.6036)
    assert not validate(_brief(evidence=ev), bundle).blocked


def test_small_integers_in_prose_are_not_treated_as_claims() -> None:
    ev = _brief().evidence
    ev[0] = ev[0].model_copy(
        update={"detail": "Only 4 of 5 comparable cases resolved the same way here."}
    )
    assert not validate(_brief(evidence=ev), _bundle()).blocked


def test_confidence_above_ninety_with_open_gaps_is_blocked() -> None:
    report = validate(
        _brief(confidence=0.95, missing_information=["device_intel: not found"]), _bundle()
    )
    assert report.blocked
    assert any(v.rule == "confident_despite_gaps" for v in report.violations)


def test_one_sided_block_is_warned_not_blocked() -> None:
    """Worth flagging, not worth refusing -- some cases genuinely have nothing
    exculpatory, and blocking the brief would lose the analyst the evidence."""
    ev = [
        e.model_copy(update={"direction": Direction.RISK_INCREASING}) for e in _brief().evidence
    ]
    report = validate(
        _brief(evidence=ev, recommendation=Recommendation.BLOCK, confidence=0.9), _bundle()
    )
    assert not report.blocked
    assert any(v.rule == "one_sided_evidence" and v.severity == "warn" for v in report.violations)


def test_grounding_counter_matches_the_validator() -> None:
    grounded, total = evidence_is_grounded(_brief(), _bundle())
    assert total >= 1
    assert grounded == total
