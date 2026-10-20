# ADR 0002 — The LLM writes briefs; it does not make the routing decision

- **Status:** accepted
- **Date:** 2026-08-13
- **Decides:** what the LLM is actually for

## Context

Having confined the LLM to the ambiguous band ([ADR 0001](0001-gbdt-in-hot-path.md)),
a second question follows: what does it do there? The tempting answer is "decide"
— read the evidence, output approve or block. That would make the agent the
decision-maker for 12% of the portfolio.

## Decision

The LLM's deliverable is a **structured brief**, not a decision. It assembles
evidence from four tools, cites what it found, states what it could not find,
and recommends. The recommendation is acted on automatically only above a
confidence floor that output policy enforces; everything else goes to a human
with the brief attached.

The value proposition is explicitly **analyst time**, not accuracy. Analysts
spend most of a review assembling context — pulling customer history, checking
the device, finding comparable past cases. That assembly is what the agent
automates. The judgement stays where it was.

## Consequences

Measured on the 150-case eval set (deterministic provider, 3 repeats):

| | value |
|---|---|
| Agreement with confirmed outcome (macro across strata) | 0.971 |
| Decisive rate (not escalated) | 0.678 |
| Schema compliance, first attempt | 1.000 |
| Numeric claims traceable to evidence | 1.000 (1,309 claims) |
| Output-policy clean rate | 1.000 |

The pairing of agreement and decisive rate is the honest reading. An agent that
escalated everything would score 1.0 agreement on an empty denominator, which is
why `decisive_rate` is a gated metric in `evals/gate.py` and why both appear in
the README table. Neither number means anything alone.

### What this buys

- Every brief is a `TriageBrief`. Downstream code branches on typed fields; no
  regex ever parses prose.
- `missing_information` gives the router a reason to escalate that is *not*
  low confidence — incomplete evidence and uncertain judgement are different
  failures and should route differently.
- A wrong brief costs an analyst thirty seconds of reading. A wrong autonomous
  decision costs a customer.

### What it costs

Two model calls' worth of latency on 12% of volume, and an escalation rate of
0.32 that a more decisive design would reduce. We consider that the right side
of the trade for a system that declines customers.

## Alternatives rejected

**Agent decides autonomously above a confidence threshold.** This is what the
system does — but only above 0.80, and above 0.85 for a block, and never with
open `missing_information`. The rejected version is the one with no floors,
where a model's self-reported 0.7 is treated as sufficient. Confidence
calibration in the eval report shows stated confidence tracking measured
accuracy well in the mid band and thinly populated above it, which is exactly
the regime where a floor is doing work.

**Agent chooses which tools to call.** Rejected for now. There are four tools,
all cheap, all relevant to every case; a planning step would add a model call
and a failure mode to save nothing. This flips the moment there are ~10 tools or
any of them is expensive.

**Free-text brief, no schema.** Rejected outright. Everything downstream —
console rendering, auto-routing, eval scoring, the groundedness check — depends
on typed fields.

## When this flips

If the confidence-calibration table shows the model well calibrated at the top
of the range across several thousand real cases, the floors could be lowered and
the decisive rate raised. That is an argument to be made with production data,
not with a 150-case offline set.
