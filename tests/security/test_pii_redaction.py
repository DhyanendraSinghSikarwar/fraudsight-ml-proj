"""No PAN, email, or national ID may reach a provider payload or a trace.

This is the control the NFR table promises. It is tested at the boundary the
promise is about -- the assembled prompt -- not at the redactor in isolation,
because a redactor that works perfectly on a field nobody passes it is not a
control.
"""

from __future__ import annotations

import json

import pytest

from fraudsight.guardrails.pii import (
    PIILeak,
    assert_clean,
    find_leaks,
    luhn_valid,
    redact,
    redact_deep,
)

# Test PANs from the public issuer test ranges. These are the numbers payment
# vendors publish for exactly this purpose; no real card is used.
TEST_PANS = [
    "4111111111111111",  # Visa
    "5555555555554444",  # Mastercard
    "378282246310005",  # Amex, 15 digits
    "6011111111111117",  # Discover
]


@pytest.mark.parametrize("pan", TEST_PANS)
def test_luhn_accepts_real_card_shapes(pan: str) -> None:
    assert luhn_valid(pan)


@pytest.mark.parametrize("pan", TEST_PANS)
@pytest.mark.parametrize("template", ["{}", "card {}", "PAN: {}.", "{} was declined"])
def test_pan_is_redacted_in_context(pan: str, template: str) -> None:
    out = redact(template.format(pan))
    assert pan not in out.text
    assert "PAN" in out.summary()


@pytest.mark.parametrize("sep", [" ", "-"])
def test_separated_pans_are_redacted(sep: str) -> None:
    """`4111 1111 1111 1111` is the form that appears in a memo field."""
    spaced = sep.join(["4111", "1111", "1111", "1111"])
    out = redact(f"customer wrote {spaced} in the note")
    assert "4111" not in out.text


def test_long_identifiers_that_are_not_pans_survive() -> None:
    """The evidence bundle is full of long numbers -- TransactionID, card1, the
    V-block. A redactor that eats them destroys the brief."""
    text = "TransactionID 2987000 card1 13926 V258 1.4 addr1 315 amount 12345678901234"
    out = redact(text)
    assert "2987000" in out.text
    assert "13926" in out.text
    assert "315" in out.text


def test_tokens_are_stable_and_non_reversible() -> None:
    """The same card must read as the same token throughout a brief, and the
    token must not contain the number."""
    a = redact("charge on 4111111111111111")
    b = redact("second charge on 4111111111111111")
    token_a = a.text.split("charge on ")[1]
    token_b = b.text.split("charge on ")[1]
    assert token_a == token_b
    assert "4111" not in token_a
    assert token_a.startswith("<PAN:")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("reach me at analyst@bank.example.com", "EMAIL"),
        ("ssn 123-45-6789 on file", "SSN"),
        ("IBAN GB29NWBK60161331926819 transfer", "IBAN"),
    ],
)
def test_other_identifiers_are_redacted(text: str, kind: str) -> None:
    out = redact(text)
    assert kind in out.summary()


def test_cvv_is_masked_without_a_token() -> None:
    out = redact("cvv: 123 supplied")
    assert "123" not in out.text
    assert "<CVV>" in out.text


def test_redact_deep_covers_keys_and_nested_values() -> None:
    payload = {
        "customer": {"email": "a@b.example.com", "cards": ["4111111111111111"]},
        "a@b.example.com": "used as a key",
    }
    cleaned, hits = redact_deep(payload)
    blob = json.dumps(cleaned)
    assert "4111111111111111" not in blob
    assert "a@b.example.com" not in blob
    assert len(hits) >= 3


def test_assert_clean_fails_closed() -> None:
    with pytest.raises(PIILeak, match="was not made"):
        assert_clean("charge on 4111111111111111", where="test payload")
    assert_clean("charge on <PAN:abcd1234>")


def test_find_leaks_reports_kinds_without_echoing_the_value() -> None:
    leaks = find_leaks("4111111111111111 and a@b.example.com")
    kinds = {leak["kind"] for leak in leaks}
    assert kinds == {"PAN", "EMAIL"}
    for leak in leaks:
        assert "4111111111111111" not in leak["sample"]


# --------------------------------------------------------------------------
# The real boundary: the assembled prompt
# --------------------------------------------------------------------------


async def test_no_pan_reaches_the_provider_payload(pipeline, eval_cases) -> None:  # type: ignore[no-untyped-def]
    """Poison the attacker-reachable free-text fields with a PAN and assert the
    prompt the provider would receive is clean.

    Poisoned fields are also model inputs, so an unseen level can shift the
    score out of the triage band and skip the LLM path entirely -- which would
    make this test pass by never exercising the thing it checks. So we search
    for a case that still lands in the band rather than asserting the first one
    does.
    """
    state = None
    for case in eval_cases[:25]:
        poisoned = dict(case.transaction)
        for field in ("DeviceInfo", "P_emaildomain", "R_emaildomain"):
            poisoned[field] = f"acct 4111111111111111 {poisoned.get(field, '')}"
        decision = await pipeline.process(poisoned)
        if decision.state is not None:
            state = decision.state
            break

    assert state is not None, "no poisoned case stayed in the triage band"

    assert "4111111111111111" not in state.user_turn
    assert not find_leaks(state.user_turn)
    assert state.redactions.get("PAN", 0) >= 1

    # And the trace, which is written to disk and shipped to a vendor UI.
    assert not find_leaks(json.dumps(state.trace(), default=str))
