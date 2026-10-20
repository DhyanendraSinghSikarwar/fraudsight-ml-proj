# ADR 0006 — Deterministic redaction is the control; Presidio is defence in depth

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** how PII is kept out of prompts and traces

## Context

This is a card system. Every prompt is assembled from six sources, shipped to a
third-party API, and written to a trace file that may be forwarded to a hosted
observability tool. "No raw PAN or PII in prompts or traces" is in the NFR table.

Presidio is the standard answer and has genuinely better recall on names and
free-text addresses.

## Decision

The **default redactor is deterministic** — regex plus a Luhn check, no
dependencies. Presidio runs only as an optional second stage (`.[pii]`), always
after the deterministic pass, never instead of it.

## Consequences

The argument is about what a *control* is. A control needs to be:

- **Assertable.** `tests/security/test_pii_redaction.py` asserts that no Luhn-
  valid 16-digit sequence survives, across four issuer test ranges and four
  surrounding contexts. You cannot write that test against a model that misses
  0.5% of the time.
- **Reviewable.** An auditor can read `_PAN_RE` and the Luhn function. Nobody
  audits a spaCy NER model.
- **Cheap.** `make demo` must not begin with a 300 MB model download.

Three design details that turned out to matter:

**Luhn, not `\d{16}`.** The evidence bundle is full of long numbers —
`TransactionID`, `card1`, the V-block. A naive digit-count redactor blanks the
evidence the brief is built from. The Luhn check is what makes the redactor
precise enough to be left on.

**Stable keyed tokens, not `[REDACTED]`.** The same card reads as the same
`<PAN:7f3c>` throughout a brief, so an analyst can follow the thread without
seeing the number. A constant placeholder destroys coreference and makes briefs
unreadable. The token is a keyed BLAKE2b digest — an unkeyed hash of a 16-digit
PAN is brute-forceable in seconds, which would make the no-PII claim false in
the only way that matters.

**Fail closed on the way out.** `assert_clean()` runs on the assembled prompt
immediately before dispatch, not only on the inbound record. A prompt built from
six sources is exactly where one unsanitised field survives.

### A bug this ADR's tests caught

The original trailing guard was `(?![\d.])`, which refused to match a PAN
followed by a full stop — so `card on file: 4111111111111111.` passed straight
through the redactor. A card number at the end of a sentence is the most natural
place to write one. The guard is now `(?!\.?\d)`, which still rejects a decimal
tail. This is the case for a control you can write tests against.

## Alternatives rejected

**Presidio as the primary.** Better recall, worse as a control: non-deterministic,
heavy, and untestable at the assertion level above.

**Redact at the prompt boundary only.** Would leave raw PII in tool results and
therefore in the traces. Redaction runs as the first node, on the raw record,
so every downstream consumer sees redacted values.

**Tokenise reversibly with a vault.** The right answer at a real issuer, and
over-engineering here. There is deliberately no de-tokenisation function: nothing
in the LLM path can turn a token back into a PAN.

## Known gaps

- Names and physical addresses in free text are not caught by the deterministic
  layer. IEEE-CIS contains none, so this is untested here — enable the Presidio
  extra before pointing this at data that does.
- Non-Latin digits and unusual separators are not handled.

Both are recorded in [docs/security.md](../security.md).
