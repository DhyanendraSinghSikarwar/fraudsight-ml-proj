# ADR 0007 — A rule-based provider ships as the fallback, the eval floor, and the CI path

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** what happens when there is no LLM, and what the LLM is measured against

## Context

Three separate problems with one shape:

1. **Graceful degradation.** The NFR says a provider outage must never drop a
   transaction. Something has to produce a brief when nothing else can.
2. **An interpretable eval number.** "The agent scores 0.89 agreement" is
   unreadable without a denominator. Compared to what?
3. **A CI gate that runs.** The gate must run on a pull request from a fork. A
   gate that needs a vendor secret gets skipped, and a skipped gate is not a gate.

## Decision

`llm/deterministic.py` implements the full `LLMClient` protocol with rules. It is
appended to the fallback ladder unconditionally and cannot be configured away.
`triage/rules.py` registers the `TriageBrief` synthesiser; `evals/judges.py`
registers rule-based judges.

It is **shipped code, not a mock**: same protocol, same schema validation, same
policy checks, scored by the same eval suite.

## Consequences

Measured over the 150-case eval set, 3 repeats:

| | rule-based floor |
|---|---|
| Agreement, macro across strata | 0.971 |
| Agreement, ambiguous stratum | 0.913 |
| Decisive rate | 0.678 |
| Schema pass, first attempt | 1.000 |
| Numeric grounding | 1.000 |
| Cost per 1,000 triages | $0.00 |
| Run-to-run std over 3 repeats | 0.000 |

Two of those deserve comment.

**The floor is strong on this eval set, and that is a finding about the eval
set as much as about the rules.** Two of the three strata are cases where the
label agrees with the score's direction, and a weighted sum of signals handles
those. The ambiguous stratum — where the label contradicts the score — is where
the gap should appear, and it is the number to watch when an LLM runs.

**Zero variance is structural.** The rules are deterministic, so three repeats
produce identical numbers. A real provider will not, which is precisely why
`--repeat 3` and the mean ± std are in the report: with an LLM, that column
stops being zero and starts telling you whether an improvement is real.

The floor's weaknesses are visible and expected: mean narrative score 3.15 out
of 5, an escalation rate of 0.32, and no capacity to weigh an unusual
*combination* of signals. It ranks pre-extracted signals and renders a template.

**Its hallucination rate is structurally zero** — every sentence is assembled
from retrieved values. That makes it a useful control: when the LLM's measured
rate is compared against it, the comparison isolates the model rather than the
harness.

## What this does NOT claim

It does not claim the LLM is unnecessary. It claims the LLM's contribution is
the *delta*, and that the delta has to be measured rather than assumed. The
run in this repository has no API key configured, so agent and floor are the
same provider and every delta is 0.0000 — which the report states plainly rather
than hiding behind an absolute score.

## Alternatives rejected

**Fail the triage when the provider is down.** Violates the NFR: a dropped
transaction is worse than a mediocre brief.

**Return the score with no brief.** An analyst then does the whole assembly by
hand, which is the cost the system exists to remove.

**A mock that returns a canned brief.** Would not survive schema validation or
policy checks and would make the CI gate meaningless.

**Skip the eval gate when no key is present.** The most common version of this
mistake, and the one that quietly turns a gate into decoration.
