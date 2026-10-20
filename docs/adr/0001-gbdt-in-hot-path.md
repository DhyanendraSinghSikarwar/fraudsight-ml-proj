# ADR 0001 — A gradient-boosted model, not an LLM, scores every transaction

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** what runs on the authorisation path

## Context

Every transaction needs a risk decision inside the authorisation window. The
budget is tens of milliseconds and the volume is the entire card portfolio. The
obvious 2026 instinct is to put a model with reasoning ability in that path.

## Decision

A LightGBM classifier with isotonic calibration scores 100% of transactions. No
LLM is on the authorisation path, under any condition, including provider
outage. The LLM is confined to the investigation layer, which sees only the
ambiguous band.

## Consequences

Measured on the held-out test split (84,233 transactions, 3.51% fraud):

| | GBDT scorer | An LLM in the same position |
|---|---|---|
| p99 latency, single transaction | **0.78 ms** | 2–6 s |
| Cost per 1,000 transactions | **$0** | ~$1–3 |
| Reproducible across runs | yes, exactly | no |
| Auditable decision | feature contributions | prose |

The AUC-PR is 0.490 ± 0.002 across three seeds. That is not a spectacular
number in absolute terms and it is not meant to be — what matters is that it
ranks well enough to make the band routing work, at a latency and cost an LLM
cannot approach by three orders of magnitude.

The calibration is what makes the score usable rather than merely ordered.
Expected calibration error is 0.0015 on test, which is what permits the score
to be fed to a cost matrix: "0.40" has to genuinely mean "40% of these are
fraud" before dollar thresholds can be derived from it.

### What we gave up

An LLM would bring world knowledge the GBDT cannot have — that a merchant is a
known reshipping front, that a device string matches a botnet family. We do get
that, but only for the 12% of volume where it can pay for itself.

## Alternatives rejected

**LLM scores everything.** At $0.0012 per call and 84k test transactions, one
pass over the test split alone costs ~$100 and takes hours. At portfolio scale
it is a seven-figure annual line item to replace a model that already ranks
better on this data.

**LLM as a re-ranker over the top 5%.** Considered seriously; it is close to
what we built. The difference is that the triage band is chosen by expected
cost rather than by a fixed percentile, which means the boundary moves when the
cost of an analyst or the price of a token moves. A fixed 5% is a number nobody
can defend when asked why.

**A neural tabular model (FT-Transformer, TabNet).** More expensive to train,
slower to serve, and on this dataset gradient boosting is still the stronger
family. Revisit if the feature space grows to include sequence structure that
trees cannot express.

## When this flips

If per-token cost falls ~1,000× **and** p99 latency reaches single-digit
milliseconds, the calculus changes and this ADR should be reopened. Neither is
close. A more plausible trigger is a fraud pattern that is genuinely legible
only in unstructured text — a memo field, a chat transcript — at which point the
right answer is probably a small specialised model in the hot path, not a
frontier one.
