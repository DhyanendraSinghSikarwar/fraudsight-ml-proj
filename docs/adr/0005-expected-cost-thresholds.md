# ADR 0005 — Score bands are chosen by expected cost, and the recall NFR is missed

- **Status:** accepted, with a published shortfall
- **Date:** 2026-08-13
- **Decides:** where the auto-approve / triage / auto-block boundaries sit

## Context

Three bands need two thresholds. The conventional approach maximises F1 or picks
a round percentile. Both implicitly price a missed fraud the same as a wrongly
declined customer, which is false by roughly seven to one on this cost matrix.

## Decision

Bands minimise **total expected cost** on the validation split, subject to a
recall floor, using an explicit cost matrix:

| Input | Value | What it is |
|---|---|---|
| `fraud_loss_usd` | 180.00 | average net loss on a missed fraud |
| `false_decline_usd` | 25.00 | churn/goodwill cost of wrongly blocking |
| `analyst_review_usd` | 4.20 | 6 min × $42/h loaded rate |
| `triage_llm_usd` | 0.0012 | measured cost of one LLM triage |
| `analyst_accuracy` | 0.92 | analysts are not oracles |

These are business inputs, not hyperparameters, and they live in config where a
risk manager can change them without touching model code.

## The shortfall, stated plainly

**The NFR of recall ≥ 0.95 at a review rate ≤ 0.15 is not achievable with this
scorer.** No `(lo, hi)` pair satisfies both. The code refuses to silently relax
the floor — `optimise()` raises — and `select_operating_point()` then ships the
cost-optimal bands with `nfr_recall_met: false` recorded in
`artifacts/thresholds.json` and surfaced in the eval report header.

The frontier, measured on validation, prices what the floor would have cost:

| Recall target | Achieved | Review rate | Cost / txn |
|---|---|---|---|
| 0.60 | 0.741 | 14.7% | $2.669 |
| 0.75 | 0.796 | 20.9% | $2.755 |
| 0.85 | 0.852 | 28.0% | $2.890 |
| 0.90 | 0.940 | 55.2% | $4.103 |
| **0.95** | **0.970** | **68.7%** | **$4.771** |
| 0.99 | 0.998 | 94.6% | $6.229 |

Holding recall at 0.95 would route **5.6× more volume to analysts** and cost
**78% more per transaction** than the chosen point. At that review rate the
system is no longer reducing analyst workload, which was its purpose.

The shipped operating point, and what it does on the untouched test split:

| | val (chosen on) | test (reported) |
|---|---|---|
| lo / hi | 0.0541 / 0.5313 | same |
| Recall | 0.716 | **0.736** |
| Review rate | 12.2% | **12.4%** |
| Auto-approve | 86.7% | 86.3% |
| Auto-block | 1.08% | 1.33% |
| Cost / txn | $2.656 | **$2.682** |

Against the baselines on test: reviewing everything costs $6.60/txn, approving
everything $5.85/txn, and a single-threshold block-only system $3.52/txn.

## Consequences

Recall is a **constraint, not a term in the objective**. Expressing it as a
penalty would let a large enough cost saving buy its way through the floor,
which in a fraud system is a commitment to the business rather than a
preference.

`lo` is searched on the quantile grid of the score distribution, not a uniform
grid — with 96% of mass below 0.10, a uniform grid spends almost every point
where nothing happens.

## What would close the recall gap

Ranking quality, not threshold tuning. Recall@5%FPR is 0.633; the frontier is a
property of the scorer. In rough order of expected value:

1. The full 339-column V-block instead of the 94-column subset (~4× memory and
   training time for ~0.003 AUC-PR measured — see `docs/scope.md`).
2. Entity-graph features — shared-device and shared-address components — which
   the flat table cannot express.
3. More history. 182 days with a 14-day embargo is a thin window for a
   drift-heavy problem.

## Alternatives rejected

**Maximise F1.** Optimises a quantity with no business meaning and prices the
two error types identically.

**Fixed percentiles ("triage the top 5%").** Cannot be defended when asked why 5,
and does not move when the price of an analyst or a token moves.

**Lower the recall floor in config until it passes.** This is the decision this
ADR exists to refuse. The floor stays at 0.95, the miss is reported, and the
gap is a roadmap item rather than a config edit.
