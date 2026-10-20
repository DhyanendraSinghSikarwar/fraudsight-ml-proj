"""The evidence bundle and the signal vocabulary derived from it.

This module is the seam between "facts the system retrieved" and "claims the
brief makes". Everything downstream reads from here:

* the prompt builder renders the bundle into the LLM's user turn
* the deterministic provider ranks `derive_signals()` output into a brief
* the groundedness judge checks every `Evidence.detail` against
  `bundle.claim_values()`
* the analyst console renders the same facts next to the model's narrative

Having one derivation is what makes the groundedness metric meaningful. If the
prompt showed the model one set of numbers and the judge checked against
another, a "hallucination rate" would be measuring the gap between two code
paths rather than anything about the model.

The signals are deliberately *named and thresholded in code*, not left for the
LLM to invent. A model asked to freestyle "what looks suspicious" produces a
different taxonomy every run, and no eval can track a moving vocabulary. The
model's job is judgement -- which signals matter, how they combine, whether the
case is ambiguous -- not signal discovery.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from fraudsight.tools.base import ToolResult

# Thresholds are policy, so they live in one visible block rather than scattered
# through the rules. Each is a round number chosen from the training
# distribution (see notebooks/01_eda.ipynb), not a tuned constant -- tuning
# these against the eval set would be fitting the evidence layer to the answers.
AMOUNT_SPIKE_RATIO = 2.0  # transaction / customer p95
VELOCITY_1H_ALERT = 3  # transactions on the same card within an hour
VELOCITY_24H_ALERT = 10
DEVICE_CARD_SHARING_ALERT = 0.5  # distinct cards per transaction on a device
MERCHANT_FRAUD_MULTIPLE = 2.0  # merchant fraud rate / portfolio rate
# Calibrated score / portfolio base rate. At a 3.4% base rate this puts the
# boundary near 0.07 -- comfortably inside the triage band, which is the point:
# a transaction only reaches triage because the score is already elevated.
LIFT_RISK_INCREASING = 2.0
ESTABLISHED_TENURE_DAYS = 30.0
NIGHT_HOURS = (0, 6)


@dataclass(frozen=True)
class Signal:
    name: str
    detail: str
    direction: str  # risk_increasing | risk_decreasing
    weight: float
    source_tool: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.name,
            "detail": self.detail,
            "direction": self.direction,
            "weight": round(self.weight, 3),
            "source_tool": self.source_tool,
        }


@dataclass
class EvidenceBundle:
    """Everything gathered for one transaction under triage."""

    transaction_id: str
    transaction: dict[str, Any]
    model: dict[str, Any]
    tools: dict[str, ToolResult] = field(default_factory=dict)
    velocity: dict[str, dict[str, Any]] = field(default_factory=dict)
    similar_cases: list[dict[str, Any]] = field(default_factory=list)
    portfolio: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    # Ratios and comparisons the evidence layer computed from retrieved values.
    # These are part of the evidence, not commentary on it: the agent is
    # expected to cite "3.6x the customer's p95", and a groundedness check that
    # only recognises the two inputs would flag the quotient as invention. The
    # alternative -- exempting derived numbers from the check -- would open the
    # exact hole the check exists to close, so they get recorded instead.
    derived: dict[str, float] = field(default_factory=dict)

    def note_derived(self, name: str, value: float, *, decimals: int = 1) -> float:
        """Record a derived quantity at the precision it will be cited at."""
        rounded = round(float(value), decimals)
        self.derived[name] = rounded
        return rounded

    # -- access helpers -------------------------------------------------

    def tool_fields(self, name: str) -> dict[str, Any]:
        r = self.tools.get(name)
        return dict(r.fields) if r is not None and r.found else {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction": self.transaction,
            "model_assessment": self.model,
            "customer_history": self.tool_fields("customer_history"),
            "device_intel": self.tool_fields("device_intel"),
            "merchant_profile": self.tool_fields("merchant_profile"),
            "velocity": self.velocity,
            "similar_cases": self.similar_cases,
            "portfolio_baseline": self.portfolio,
            "derived_comparisons": self.derived,
            "unavailable_evidence": self.missing,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)

    def claim_values(self) -> set[str]:
        """Every value an `Evidence.detail` may cite. The groundedness check
        looks for numeric tokens in a detail string and asks whether each one
        appears here."""
        from fraudsight.tools.base import _renderings, _walk

        out: set[str] = set()
        for r in self.tools.values():
            out |= r.claim_values()
        for source in (self.transaction, self.model, self.velocity, self.portfolio, self.derived):
            for v in _walk(source):
                out |= _renderings(v)
        for case in self.similar_cases:
            for v in _walk(case):
                out |= _renderings(v)
        return out


# --------------------------------------------------------------------------
# Signal derivation
# --------------------------------------------------------------------------


def attach_score_lift(bundle: EvidenceBundle) -> tuple[float, float]:
    """Compute the score's lift over the portfolio base rate and *store it on
    the bundle*.

    Storing rather than only returning is the point. Groundedness is checked by
    asking whether every number in the brief appears in the evidence, so a
    derived quantity the agent is expected to cite has to become part of the
    evidence -- otherwise the guardrail correctly flags an honest claim as a
    hallucination. The alternative, exempting derived numbers from the check,
    would open exactly the hole the check exists to close.
    """
    base = float(bundle.portfolio.get("portfolio_fraud_rate") or 0.034) or 0.034
    score = _f(bundle.model.get("score"))
    lift = round(score / base, 1) if base > 0 else 0.0
    bundle.model["portfolio_fraud_rate"] = round(base, 3)
    bundle.model["score_vs_base_rate"] = lift
    return round(base, 3), lift


def derive_signals(bundle: EvidenceBundle) -> list[Signal]:
    """Rule-derived signals, strongest first.

    Every `detail` string states a retrieved number so the claim is checkable.
    Weights are fixed per rule rather than learned: they order the brief, they
    do not decide the outcome, and a learned weight here would be a second
    uncalibrated model hiding inside the evidence layer.
    """
    txn = bundle.transaction
    cust = bundle.tool_fields("customer_history")
    dev = bundle.tool_fields("device_intel")
    merch = bundle.tool_fields("merchant_profile")
    card_vel = bundle.velocity.get("card1", {})
    out: list[Signal] = []

    amount = _f(txn.get("amount"))
    score = _f(bundle.model.get("score"))

    # -- amount ---------------------------------------------------------
    p95 = _f(cust.get("amount_p95"))
    if amount and p95 and amount > p95 * AMOUNT_SPIKE_RATIO:
        ratio = bundle.note_derived("amount_vs_customer_p95", amount / p95)
        out.append(
            Signal(
                "amount_above_customer_baseline",
                f"Transaction of ${amount:,.2f} is {ratio:.1f}x this card's "
                f"historical 95th-percentile amount of ${p95:,.2f}.",
                "risk_increasing",
                0.8,
                "customer_history",
            )
        )
    elif amount and p95 and amount <= p95:
        out.append(
            Signal(
                "amount_within_customer_baseline",
                f"Transaction of ${amount:,.2f} sits within this card's usual range "
                f"(95th percentile ${p95:,.2f}).",
                "risk_decreasing",
                0.45,
                "customer_history",
            )
        )

    if txn.get("amount_is_round") and amount and amount <= 5.0:
        out.append(
            Signal(
                "card_testing_amount",
                f"Amount of ${amount:,.2f} is small and exactly round, the classic "
                f"card-testing authorisation pattern.",
                "risk_increasing",
                0.6,
                "transaction",
            )
        )

    # -- customer history ------------------------------------------------
    if not cust:
        pass
    elif cust.get("thin_history"):
        out.append(
            Signal(
                "thin_customer_history",
                f"Only {cust.get('transactions')} prior transactions on this card, too few "
                f"to establish a behavioural baseline.",
                "risk_increasing",
                0.55,
                "customer_history",
            )
        )
    else:
        tenure = _f(cust.get("tenure_days"))
        if tenure >= ESTABLISHED_TENURE_DAYS:
            out.append(
                Signal(
                    "established_customer",
                    f"Card has {cust.get('transactions')} transactions across "
                    f"{tenure:.0f} days of history.",
                    "risk_decreasing",
                    0.5,
                    "customer_history",
                )
            )

    prior_fraud = _i(cust.get("prior_confirmed_fraud"))
    if prior_fraud > 0:
        out.append(
            Signal(
                "prior_confirmed_fraud_on_card",
                f"This card has {prior_fraud} previously confirmed fraudulent transactions "
                f"({_f(cust.get('prior_fraud_rate')) * 100:.1f}% of its history).",
                "risk_increasing",
                0.85,
                "customer_history",
            )
        )

    addrs = _i(cust.get("distinct_addresses"))
    if addrs >= 5:
        out.append(
            Signal(
                "address_churn",
                f"Card has been used with {addrs} distinct billing addresses.",
                "risk_increasing",
                0.5,
                "customer_history",
            )
        )

    # -- device ----------------------------------------------------------
    if dev:
        sharing = _f(dev.get("cards_per_transaction"))
        if sharing >= DEVICE_CARD_SHARING_ALERT and _i(dev.get("distinct_cards")) > 2:
            out.append(
                Signal(
                    "device_card_sharing",
                    f"Device has been seen with {dev.get('distinct_cards')} distinct cards "
                    f"across {dev.get('transactions')} transactions.",
                    "risk_increasing",
                    0.7,
                    "device_intel",
                )
            )
        if dev.get("thin_history"):
            out.append(
                Signal(
                    "device_unseen",
                    f"Device fingerprint appears only {dev.get('transactions')} times in "
                    f"history.",
                    "risk_increasing",
                    0.45,
                    "device_intel",
                )
            )
    if _f(txn.get("has_identity_record")) == 0.0:
        out.append(
            Signal(
                "no_device_record",
                "No device or identity record was attached to this authorisation, so device "
                "reputation could not be assessed.",
                "risk_increasing",
                0.3,
                "transaction",
            )
        )

    # -- merchant --------------------------------------------------------
    if merch:
        multiple = _f(merch.get("fraud_rate_vs_portfolio"))
        if multiple >= MERCHANT_FRAUD_MULTIPLE:
            out.append(
                Signal(
                    "high_risk_merchant",
                    f"Merchant proxy {merch.get('merchant_key')} runs a fraud rate "
                    f"{multiple:.1f}x the portfolio average.",
                    "risk_increasing",
                    0.6,
                    "merchant_profile",
                )
            )
        elif multiple and multiple < 0.5:
            out.append(
                Signal(
                    "low_risk_merchant",
                    f"Merchant proxy {merch.get('merchant_key')} runs a fraud rate "
                    f"{multiple:.1f}x the portfolio average.",
                    "risk_decreasing",
                    0.35,
                    "merchant_profile",
                )
            )

    # -- velocity --------------------------------------------------------
    v1h = _i(card_vel.get("txn_last_1h"))
    v24h = _i(card_vel.get("txn_last_24h"))
    if v1h >= VELOCITY_1H_ALERT:
        out.append(
            Signal(
                "velocity_spike",
                # Windows are named in words rather than numerals ("the
                # preceding day", not "24 hours"). The window size is a label,
                # not a measurement, and rendering it as a bare numeral invites
                # every downstream reader -- including the groundedness check --
                # to treat it as a claim.
                f"{v1h} transactions on this card in the preceding hour "
                f"({v24h} across the preceding day).",
                "risk_increasing",
                0.75,
                # Attributed to `transaction`, not `customer_history`: velocity
                # comes from the streaming feature store, which is part of the
                # transaction pipeline. Attributing it to a tool that may have
                # returned nothing would trip the misattribution policy rule --
                # correctly, because it would be a false citation.
                "transaction",
            )
        )
    elif v24h >= VELOCITY_24H_ALERT:
        out.append(
            Signal(
                "elevated_daily_velocity",
                f"{v24h} transactions on this card across the preceding day.",
                "risk_increasing",
                0.5,
                "transaction",
            )
        )

    # -- transaction-level -----------------------------------------------
    hour = _f(txn.get("hour_of_day"))
    if NIGHT_HOURS[0] <= hour < NIGHT_HOURS[1]:
        out.append(
            Signal(
                "overnight_transaction",
                f"Authorised at {int(hour):02d}:00 local, when a genuine cardholder is least "
                f"likely to notice a notification.",
                "risk_increasing",
                0.3,
                "transaction",
            )
        )

    match = _f(txn.get("email_domains_match"), default=-1.0)
    if match == 0.0:
        out.append(
            Signal(
                "email_domain_mismatch",
                f"Purchaser domain {txn.get('purchaser_email_domain')} differs from recipient "
                f"domain {txn.get('recipient_email_domain')}.",
                "risk_increasing",
                0.4,
                "transaction",
            )
        )

    # -- similar cases ---------------------------------------------------
    if bundle.similar_cases:
        fraud_like = sum(1 for c in bundle.similar_cases if c.get("outcome") == "confirmed_fraud")
        total = len(bundle.similar_cases)
        if fraud_like > total / 2:
            out.append(
                Signal(
                    "similar_to_confirmed_fraud",
                    f"{fraud_like} of {total} retrieved comparable cases were confirmed fraud.",
                    "risk_increasing",
                    0.65,
                    "similar_cases",
                )
            )
        elif fraud_like < total / 2:
            out.append(
                Signal(
                    "similar_to_legitimate_activity",
                    f"{total - fraud_like} of {total} retrieved comparable cases were confirmed "
                    f"legitimate.",
                    "risk_decreasing",
                    0.5,
                    "similar_cases",
                )
            )

    # -- the model itself -------------------------------------------------
    # Judged against the portfolio base rate, not against 0.5. At a 3.4% base
    # rate a calibrated 0.40 means "twelve times more likely to be fraud than a
    # random transaction", which is emphatically risk-increasing; comparing it
    # to a coin flip would file it as exculpatory and invert the whole brief.
    #
    # Skipped entirely for corpus entries: a resolved historical case has no
    # live score, and emitting `model_risk_score` on all 1,500 of them would put
    # a constant token in every corpus document, which is precisely the kind of
    # term that makes a lexical retriever match everything equally.
    if bundle.model.get("band") == "corpus":
        out.sort(key=lambda s: -s.weight)
        return out

    base, lift = attach_score_lift(bundle)
    increasing = lift >= LIFT_RISK_INCREASING
    out.append(
        Signal(
            "model_risk_score",
            f"The calibrated scorer put this transaction at {score:.3f}, "
            f"{lift:.1f}x the portfolio fraud rate of {base:.3f}, in the "
            f"{bundle.model.get('band')} band.",
            "risk_increasing" if increasing else "risk_decreasing",
            # Log-scaled: the jump from 1x to 4x the base rate is meaningful,
            # the jump from 20x to 23x is not.
            round(min(0.9, 0.25 + 0.22 * math.log1p(max(lift, 0.0))), 3)
            if increasing
            else round(min(0.6, 0.25 + 0.35 * (1.0 - min(lift, 1.0))), 3),
            "transaction",
        )
    )

    out.sort(key=lambda s: -s.weight)
    return out


def signal_names(signals: list[Signal]) -> list[str]:
    return [s.name for s in signals]


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
