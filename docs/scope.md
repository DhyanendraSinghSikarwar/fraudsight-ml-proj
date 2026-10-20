# What this system does not do, and why

Showing what was rejected is a stronger signal than showing what was built.
Everything below is a deliberate decision with a measurement or an argument
behind it, and a condition under which it should be revisited.

---

## 1. Targets that are missed

### Recall is 0.736, not the 0.95 in the NFR table

The single largest gap. Fully documented in
[ADR 0005](adr/0005-expected-cost-thresholds.md) with the measured frontier.

Short version: no threshold pair reaches 0.95 recall at a defensible review
rate on this scorer. Holding the floor would route 68.7% of transactions to
analysts at $4.77/txn instead of 12.4% at $2.68/txn — 5.6× the review volume and
78% more cost, which defeats the system's purpose. The code refuses to silently
lower the floor: it ships the cost-optimal point and stamps
`nfr_recall_met: false` on the artifact and the eval report header.

**What would close it:** ranking quality, not tuning. Recall@5%FPR is 0.633.

### Throughput is not 2,000 tps in one process

Measured end-to-end on a 40,000-transaction replay: **487 tps** sustained in one
Python worker, with scoring latency of 1.09 ms p50 / 6.14 ms p99. The 2,000 tps
target needs four or five workers behind a load balancer — a deployment decision
this repository does not make.

This is measured wall-clock over the whole replay, not extrapolated from a batch
benchmark or from per-call latency. **What is not claimed: that one process hits
2,000 tps.**

The first honest measurement was 16 tps. The bottleneck was not the model — it
was the feature store rescanning every event in each sliding window on every
lookup, which is O(n²) over a replay and cost ~58 ms per transaction against
~3.6 ms of scoring. Two fixes: running totals with a per-window head pointer
(amortised O(1), verified against the naive implementation over 3,000 randomised
steps), and treating the `__missing__` sentinel as "no entity" rather than as one
entity that every null-keyed transaction shares. The second was a correctness bug
as much as a performance one — it was reporting velocities in the thousands for
"the missing address", which the agent would have cited as a spike.

### Judge calibration against human labels is not done

The blueprint asks for Cohen's kappa against 30 human labels. The plumbing is
complete — `fraudsight calibrate`, the interactive labeller, and
`evals/labelling_protocol.md` — but no human has labelled the set. The shipped
`reference_labels.jsonl` is rule-based, so kappa is reported as `judge_vs_rules`
with `independent: false`, and the eval gate **refuses to gate on any
judge-derived metric** until real labels exist.

Consequence, stated plainly: **the hallucination rate and narrative score in the
README are unvalidated.** The mechanical metrics (numeric grounding, schema
pass, policy clean rate) do not depend on the judge and are gated normally.

**What would close it:** about 90 minutes with the protocol.

---

## 2. Deliberately not built

### No graph features

Shared-device and shared-address connected components are the classic next
feature family for card fraud and would very likely help recall most. Not built
because they need a graph store on the serving path and a per-transaction
neighbourhood query inside a 50ms budget. Cost is real; benefit is unmeasured
here. **Revisit when** the recall gap becomes the binding constraint — this is
the first thing to try.

### Only 94 of 339 V-columns

The full V-block costs roughly 4× the memory and training time. Measured on this
data, the curated subset gives up about 0.003 AUC-PR, which is inside the
run-to-run spread of 0.002. Keeping 94 columns chosen to span the correlation
clusters is the better trade at this scale. **Revisit when** training moves off
a laptop.

### Velocity counters are not model features

The feature store computes card/device/address velocity, and the triage agent
cites it — but the scorer never sees it. IEEE-CIS is a flat table with no
session context, so a model trained on it never saw a velocity feature, and
supplying one at serve time would be textbook train/serve skew.
`tests/unit/test_online_features.py::test_velocity_is_not_a_model_feature`
enforces the separation. **Revisit when** training data comes from the same
stream the serving path sees.

### The embargo is 14 days, not 45

The honest chargeback maturity window for a mixed portfolio is ~45 days.
IEEE-CIS spans 182. A 45-day embargo on a 60% training window leaves ~28k rows
and ~1k positives per eval split — small enough that run-to-run noise would
swamp any real improvement. 14 days preserves ~30-day val and test windows.

**This makes the offline numbers slightly optimistic.** Set `embargo_days=45` on
a multi-year internal history.

### Merchant profiles are a proxy

IEEE-CIS ships no merchant column. `ProductCD × addr1` is the closest available
stand-in and every merchant tool result carries a `basis` field saying so, which
propagates into the brief. A high fraud multiple is a segment-level signal, not
a claim about a named business.

### Device and merchant profile granularity inherits the vocabulary cap

Categorical features are capped at 200 levels, so `DeviceInfo` collapses to 201
distinct profiles and merchants to 500. Rare devices share the `__other__`
profile. Acceptable for evidence; would need lifting before device reputation
carried real weight.

### No fine-tuning

~57% of teams do not fine-tune, and this system has no evidence it would help.
The LLM's job is narrative synthesis over structured evidence — a task base
models do well. **Revisit when** there is a high-volume, narrow, measurable task
and an eval set showing a base model failing it.

### No streaming to the console

Briefs are 3–5 sentences produced in one call. Streaming would add complexity
for perhaps a second of perceived latency on a screen the analyst reads for
fifteen. **Revisit if** briefs get longer or an interactive follow-up is added.

### Single-worker deployment

No shared circuit-breaker state, no distributed budget counter, deterministic
backoff with no jitter. All three are correct at one worker and wrong at twenty.

---

## 3. Known measurement limitations

**The agent-vs-floor delta in this run is 0.0000 on every metric**, because no
API key was configured, so both sides ran the same deterministic provider. The
eval report says so in its header (`provider: deterministic`). The comparison
machinery is real and tested; the comparison itself is pending a key.

**Run-to-run variance is 0.000 across 3 repeats**, which is structural: the
rule-based provider is deterministic. With a real provider that column becomes
informative — and it is the column that tells you whether an improvement is real.

**The "ambiguous" stratum is defined mechanically**, not by observed analyst
disagreement, which this dataset does not record. A case is ambiguous when the
label contradicts the direction the score leans. The definition is published in
`data/cases.py` rather than implied.

**The corpus over-samples fraud (40% vs a 3.4% base rate)** so that retrieval
returns useful neighbours for fraud patterns. The tool reports
`confirmed_fraud_among_returned` rather than letting the agent read the mix as a
population rate.

---

## 4. Things that would be wrong to add

**An LLM in the authorisation path.** [ADR 0001](adr/0001-gbdt-in-hot-path.md).

**Autonomous blocking below the confidence floors.** The floors exist because a
false decline costs a customer relationship.

**A "clean sweep" security claim.** The red team blocks 40/40, but the scanner
only matched 31 of those 40 payloads — nine were stopped by structural
separation and the output schema. That gap is published in
[docs/security.md](security.md) because a pattern matcher will miss novel
phrasings, and claiming otherwise would be the least trustworthy sentence in
this repository.
