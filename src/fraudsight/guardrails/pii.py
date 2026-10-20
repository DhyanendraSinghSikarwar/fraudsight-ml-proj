"""PII redaction. Runs before any text reaches a provider or a trace.

Design decisions that matter more than the regexes:

* **Deterministic first, ML second.** The default redactor is pure regex plus a
  Luhn check and has no dependencies. Presidio is a genuinely better *recall*
  engine for names and free-text addresses, but it is a 300MB spaCy download
  that would make `make demo` a ten-minute affair, and -- more importantly --
  an ML redactor that misses a PAN 0.5% of the time is not a control you can
  write a security test against. The deterministic layer is the control; the
  optional Presidio pass (`.[pii]`) is defence in depth on free text.

* **Redaction is not masking-in-place.** Every hit is replaced with a stable
  token derived from a keyed hash of the value, so the same card reads as the
  same `<PAN:7f3c>` across an entire brief and an analyst can follow the thread
  without ever seeing the number. Replacing with a constant `[REDACTED]`
  destroys that coreference and makes briefs unreadable.

* **The tokens are not reversible.** There is no de-tokenisation function here
  on purpose. The console joins back to the real record by transaction ID from
  its own database; nothing in the LLM path can turn a token back into a PAN.

* **Fail closed on the way out, too.** `assert_clean()` is called on the
  outbound provider payload, not only on the inbound evidence. A prompt
  assembled from six sources is exactly where one unsanitised field survives,
  and tests/security/test_no_pan_leak.py runs it over every eval case.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# A keyed hash, not a bare digest: an unkeyed hash of a 16-digit PAN is
# brute-forceable in seconds, which would make the "traces contain no PII" claim
# false in the only way that matters.
_TOKEN_SALT = b"fraudsight-redaction-v1"


class PIILeak(RuntimeError):
    """Raised when unredacted PII is found on an outbound payload."""


@dataclass
class Redaction:
    kind: str
    token: str
    count: int = 1


@dataclass
class RedactionReport:
    text: str
    redactions: list[Redaction] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.redactions

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.redactions:
            out[r.kind] = out.get(r.kind, 0) + r.count
        return out


def _token(kind: str, value: str) -> str:
    digest = hashlib.blake2b(
        value.encode("utf-8"), key=_TOKEN_SALT, digest_size=4
    ).hexdigest()
    return f"<{kind}:{digest}>"


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

# 13-19 digits with optional separators. The Luhn check is what keeps this from
# eating every long identifier in the payload -- TransactionID, card1 and the
# V-columns are all numeric, and a naive \d{16} redactor would blank out the
# evidence the brief is built from.
#
# The trailing guard is `(?!\.?\d)`, not `(?![\d.])`. The stricter form refused
# to match a PAN at the end of a sentence -- "card on file: 4111111111111111."
# -- because the full stop tripped the lookahead, which meant a card number in
# the most natural place to write one sailed straight through the redactor. The
# guard still has to reject a decimal tail so that 12345678901234.56 is read as
# one number rather than as a PAN, hence `\.?\d` rather than a bare `\d`.
_PAN_RE = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?!\.?\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\d.])\+?\d{1,3}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\d.])")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CVV_RE = re.compile(r"\b(?:cvv|cvc|cv2)\s*[:=]?\s*(\d{3,4})\b", re.IGNORECASE)


def luhn_valid(digits: str) -> bool:
    ds = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(ds) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> RedactionReport:
    """Redact a single string. Order matters: PAN before phone, because a
    16-digit PAN written with spaces also matches the phone pattern and the
    weaker label would win."""
    hits: list[Redaction] = []

    def sub(pattern: re.Pattern[str], kind: str, body: str, *, validator: Any = None) -> str:
        def repl(m: re.Match[str]) -> str:
            value = m.group(0)
            if validator is not None and not validator(value):
                return value
            hits.append(Redaction(kind, _token(kind, re.sub(r"\D", "", value) or value)))
            return _token(kind, re.sub(r"\D", "", value) or value)

        return pattern.sub(repl, body)

    out = text
    out = sub(_PAN_RE, "PAN", out, validator=luhn_valid)
    out = sub(_IBAN_RE, "IBAN", out)
    out = sub(_SSN_RE, "SSN", out)
    out = sub(_EMAIL_RE, "EMAIL", out)
    out = sub(_PHONE_RE, "PHONE", out)
    out = _CVV_RE.sub(lambda m: m.group(0).replace(m.group(1), "<CVV>"), out)
    return RedactionReport(text=out, redactions=hits)


def redact_deep(node: Any) -> tuple[Any, list[Redaction]]:
    """Redact every string inside a nested structure.

    Keys are redacted as well as values: an evidence bundle keyed by email
    domain would otherwise leak through the key side of the mapping.
    """
    hits: list[Redaction] = []

    def walk(n: Any) -> Any:
        if isinstance(n, str):
            r = redact(n)
            hits.extend(r.redactions)
            return r.text
        if isinstance(n, dict):
            return {walk(k): walk(v) for k, v in n.items()}
        if isinstance(n, (list, tuple)):
            return type(n)(walk(v) for v in n)
        return n

    return walk(node), hits


def find_leaks(text: str) -> list[dict[str, str]]:
    """Everything a redaction pass should have caught. Used by the security
    tests and by `assert_clean`."""
    out: list[dict[str, str]] = []
    for m in _PAN_RE.finditer(text):
        if luhn_valid(m.group(0)):
            out.append({"kind": "PAN", "sample": _elide(m.group(0))})
    for kind, rx in (("EMAIL", _EMAIL_RE), ("SSN", _SSN_RE), ("IBAN", _IBAN_RE)):
        for m in rx.finditer(text):
            out.append({"kind": kind, "sample": _elide(m.group(0))})
    return out


def assert_clean(payload: Any, *, where: str = "provider payload") -> None:
    """Fail closed. Called on the assembled prompt immediately before dispatch."""
    text = payload if isinstance(payload, str) else str(payload)
    leaks = find_leaks(text)
    if leaks:
        kinds = sorted({leak["kind"] for leak in leaks})
        raise PIILeak(
            f"{len(leaks)} unredacted PII value(s) of type {kinds} found in {where}. "
            f"The call was not made."
        )


def _elide(value: str) -> str:
    return f"{value[:2]}...{value[-2:]}" if len(value) > 6 else "..."


# --------------------------------------------------------------------------
# Optional second stage
# --------------------------------------------------------------------------


def presidio_redact(text: str) -> RedactionReport:  # pragma: no cover - optional extra
    """Names, locations, and free-text PII the regexes cannot express.

    Only ever runs *after* the deterministic pass, never instead of it.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return redact(text)

    first = redact(text)
    analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()
    results = analyzer.analyze(text=first.text, language="en")
    anonymised = anonymizer.anonymize(text=first.text, analyzer_results=results)
    extra = [Redaction(r.entity_type, "<PII>") for r in results]
    return RedactionReport(text=anonymised.text, redactions=[*first.redactions, *extra])
