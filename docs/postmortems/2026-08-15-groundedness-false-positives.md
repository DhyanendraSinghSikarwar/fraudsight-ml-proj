# Postmortem — the groundedness check spent a day measuring itself

- **Date:** 2026-08-15
- **Severity:** high (a headline quality metric was wrong in both directions)
- **Detected by:** the eval suite and the replay harness disagreeing
- **Status:** resolved

---

## Summary

`guardrails/output.py` blocks any brief citing a number that does not appear in
the evidence. It is the mechanical half of the hallucination metric and the
strictest rule in the policy layer.

Over four iterations it produced false positives at rates between 15% and 70%.
Every one was the checker misreading its own inputs. The metric it reported was
not "how often does the model invent numbers" but "how often does my regex
misparse English", and for a while the two were indistinguishable from the
outside — the rate was non-zero and plausible, the failures looked like
hallucinations, and the repair loop dutifully burned two extra model calls
trying to fix sentences that were already correct.

## Timeline

**Run 1 — 70% of briefs blocked.** Cause: `"95th percentile"`. The extractor
read `95` as a claim. Ordinals are labels.

**Run 2 — 15% blocked.** Cause: `1.5385` rendered as `"1.5%"`. A relative
tolerance of 1.1% is tighter than one decimal place of display precision, so
honest rounding read as invention.

**Run 3 — one case blocked.** Cause: `"the card's p95"`. The identifier `p95`
contains a digit; the extractor read `95` again, this time glued to a letter.

**Run 4 — 0% on the eval set, 59% on a replay.** The discrepancy was the useful
part. Cause: `"(16 in 24 hours)"`. The `24` is the size of the measurement
window, not a measurement. It only appeared in the replay because the eval set
runs without a warmed feature store, so velocity signals never fired there.

## Root cause

One mistake in four costumes: **the checker treated every numeral in a sentence
as a factual claim.** Natural language is full of numerals that are not claims —
ordinals, unit labels, window sizes, identifiers containing digits, and the
display-rounded form of a value stored at higher precision.

The deeper error was in what the fixes assumed. The first three were narrowing
the extractor, which is right. The fourth is different: `24` genuinely *was* a
number the brief asserted, and it genuinely *was not* in the evidence, because
the window size lived in a dictionary **key** (`txn_last_24h`) rather than a
value.

## What made it hard to see

The eval suite reported a clean 0% and the replay reported 59%, on the same
code, the same day. Chasing that gap is what found the bug. Two things enabled
it:

1. **The eval set does not warm the feature store.** Velocity signals are
   therefore never exercised offline — an eval-set coverage gap that hid an
   entire signal family.
2. **A plausible failure rate is the most dangerous kind.** 15% ungrounded
   claims looks like a model with a rounding problem. Had it been 100% or 0%
   somebody would have looked on day one.

## Fixes

| | Change |
|---|---|
| Ordinals | `(?!(?:st\|nd\|rd\|th)\b)` — `95th` is not a claim of 95 |
| Identifiers | `(?<![A-Za-z0-9_])` — `p95`, `V258`, `card1` are names |
| Display precision | Claims carry their decimal count; a value that rounds to the claim at the claimed precision is grounded |
| Derived ratios | `EvidenceBundle.note_derived()` — a ratio the system computes and expects to be cited becomes part of the evidence |
| Window sizes | `velocity_summary()` emits `window_hours` / `window_days` as values |
| Prose | Templates say "the preceding day", not "24 hours" |

Each fix has a regression test in `tests/unit/test_output_policy.py`.

## The rule that came out of it

> **If the system expects a brief to cite a number, that number must be in the
> evidence — not in a key, not implied by two other numbers, not in prose.**

The rejected alternative was to exempt derived numbers from the check. That
would have made the false positives disappear and opened exactly the hole the
check exists to close: an invented figure is also a number that is not in the
evidence.

## What we are not fixing

The extractor still cannot distinguish `"109 days of history"` (a claim) from
`"24 hours"` (a label) in general. The current position is that the *system's
own templates* avoid bare unit numerals, and that any window an LLM might cite
is present as a value. A model that writes `"in the last 72 hours"` about a
72-hour window we do not compute would still be flagged — correctly, since we
never measured that.

## Follow-up

- [x] Regression test per fix
- [x] Replay clears traces so the daily report is not aggregating history
- [ ] Warm the feature store in the eval runner so velocity signals are
      exercised offline (the coverage gap that hid this)
- [ ] Report the ungrounded-claim rate split by signal name, so a systematic
      failure in one template is visible rather than averaged away
