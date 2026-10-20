# fraudsight

**Real-time fraud triage.** A gradient-boosted model scores every transaction in
the hot path at 0.78 ms p99. An LLM agent investigates only the ambiguous 12%
and writes the analyst's brief.

The point of the architecture is what it *doesn't* do: 86% of transactions never
reach a language model, because a calibrated GBDT already answers them for
nothing in under a millisecond.

---

## Results

Measured, not estimated. Every number below is reproduced by `make artifacts &&
make eval`, and the commands that produce each are named.

### The scorer — held-out test split, 84,233 transactions, 3.51% fraud

| Metric | Mean | Std (3 seeds) |
|---|---|---|
| AUC-PR | **0.490** | 0.002 |
| ROC-AUC | 0.885 | 0.000 |
| Recall @ 1% FPR | 0.443 | 0.023 |
| Recall @ 5% FPR | 0.633 | 0.004 |
| Brier score | 0.023 | 0.000 |
| **Expected calibration error** | **0.0015** | 0.0001 |
| p99 single-transaction latency | **0.78 ms** | — |

ROC-AUC is shown for comparability and ignored for decisions: at a 3.5% base
rate it flatters everything. Calibration is the load-bearing number — the bands
below are dollar thresholds, and a cost matrix applied to uncalibrated scores is
meaningless.

`make train`

### Routing — bands chosen by expected-cost minimisation

| | val (chosen on) | test (reported) |
|---|---|---|
| Auto-approve (< 0.054) | 86.7% | **86.3%** |
| Triage (0.054 – 0.531) | 12.2% | **12.4%** |
| Auto-block (≥ 0.531) | 1.1% | **1.3%** |
| Recall | 0.716 | **0.736** |
| Cost per transaction | $2.656 | **$2.682** |

Against the baselines on test: reviewing everything costs **$6.60/txn**,
approving everything **$5.85/txn**, and a single-threshold system with no triage
band **$3.52/txn**. The triage band pays for itself by **24%** against the best
no-LLM alternative.

`make thresholds`

### ⚠ The recall NFR is missed, and here is what meeting it would cost

Target was recall ≥ 0.95. Achieved is **0.736**. The code refuses to silently
lower the floor — it ships the cost-optimal point and stamps
`nfr_recall_met: false` on the artifact and every eval report.

| Recall target | Achieved | Review rate | Cost / txn |
|---|---|---|---|
| 0.75 | 0.796 | 20.9% | $2.755 |
| 0.85 | 0.852 | 28.0% | $2.890 |
| 0.90 | 0.940 | 55.2% | $4.103 |
| **0.95** | **0.970** | **68.7%** | **$4.771** |

Holding the floor would send **5.6× more volume to analysts at 78% higher cost**
— the system would stop reducing review workload, which was its purpose. Full
reasoning in [ADR 0005](docs/adr/0005-expected-cost-thresholds.md); what would
close the gap is in [docs/scope.md](docs/scope.md).

### The triage agent — 150 labelled cases, 3 repeats

| Metric | Agent | Rule-based floor | Δ |
|---|---|---|---|
| Agreement with outcome (macro) | 0.986 | 0.986 | 0.000 |
| — confirmed fraud (n=50) | 1.000 | 1.000 | 0.000 |
| — confirmed legitimate (n=50) | 1.000 | 1.000 | 0.000 |
| — **ambiguous (n=49)** | **0.958** | 0.958 | 0.000 |
| Decisive rate (not escalated) | 0.758 | 0.758 | 0.000 |
| Schema pass, first attempt | 1.000 | 1.000 | — |
| Numeric claims grounded (1,309 checked) | 1.000 | 1.000 | — |
| Output-policy clean rate | 1.000 | 1.000 | — |
| Narrative score (1–5) | 4.09 | 4.09 | 0.000 |
| Run-to-run std over 3 repeats | 0.000 | 0.000 | — |

**Every delta is 0.000 because no API key was configured for this run**, so the
"agent" and the "floor" were the same rule-based provider. The report header says
so (`provider: deterministic`). Set `ANTHROPIC_API_KEY` and
`PRIMARY_PROVIDER=anthropic` and the deltas become real — the ambiguous stratum
is the row to watch, since that is where a weighted sum of signals should run
out of road.

Read agreement and decisive rate **together**. An agent that escalated everything
would score 1.0 agreement on an empty denominator, which is why `decisive_rate`
is a gated metric.

`make eval`

### The live path — replayed in event-time order

Scoring path alone, 40,000 transactions:

| | Measured |
|---|---|
| Auto-approve / triage / auto-block | **86.3% / 12.5% / 1.2%** |
| Recall on the replayed window | 0.732 |
| Scoring latency p50 / p95 / p99 | **1.09 / 2.84 / 6.14 ms** (NFR: < 50 ms p99) |
| Sustained throughput, one worker | **487 tps** |

With triage running in the same process, 4,000 transactions:

| | Measured |
|---|---|
| Triage service time p50 / p95 | 1,568 ms / **4,299 ms** (NFR: < 6 s p95) |
| Schema pass, first attempt | **100%** |
| Repairs per brief | **0.00** |
| Output-policy blocks | **0%** |

487 tps is one Python worker, so the 2,000 tps target needs four or five behind
a load balancer — a deployment decision this repo does not make, and stated
rather than extrapolated from a batch benchmark.

That number was **16 tps** until the feature store was profiled. Sliding-window
velocity counters were rescanning every event in the window on every lookup —
O(n²) over a replay, ~58 ms per transaction against ~3.6 ms of actual scoring.
Running totals with a per-window head pointer made it 580× faster, verified
against the naive implementation over 3,000 randomised steps
(`test_incremental_windows_match_a_full_rescan`).

The daily report flags **PSI 0.33** on the score distribution between the val
and test windows. That is genuine temporal drift 30 days apart — the detector
working, not a false alarm.

`make throughput && make replay && make report`

### Security — 40 injection attempts through the live pipeline

| Category | Blocked | Matched by the scanner |
|---|---|---|
| instruction_override | 8/8 | 7 |
| role_hijack | 8/8 | 6 |
| decision_steering | 8/8 | 8 |
| exfiltration | 8/8 | 6 |
| delimiter_escape | 8/8 | 4 |
| **Total** | **40/40 (100%)** | **31/40 (78%)** |

The second column matters more than the first. **Nine payloads were not matched
by the scanner** — they were stopped by structural separation and the forced
output schema. And the 100% was measured against a rule-based provider that does
not read instructions at all; against a real LLM it will be lower. Both caveats,
and five more known gaps, are in [docs/security.md](docs/security.md).

`make redteam`

### What is not validated

The LLM judge has **not** been calibrated against human labels. The plumbing is
complete (`fraudsight calibrate`, [evals/labelling_protocol.md](evals/labelling_protocol.md))
but nobody has spent the 90 minutes. So kappa is reported as `judge_vs_rules`
with `independent: false`, and **the eval gate refuses to gate on any
judge-derived metric** — the hallucination rate and narrative score above are
reported and explicitly unvalidated. The mechanical metrics do not depend on the
judge and are gated normally.

---

## Run it

No API key, no Docker, no services:

```bash
uv sync --all-extras   # the extras matter: without them there is no API and no console
make data              # fetch IEEE-CIS + PaySim  (~1.2 GB)
make artifacts         # features -> train -> thresholds -> profiles -> corpus -> eval set
make demo              # health, one triage end to end, a replay, the daily report
```

Then `make console` for the analyst UI, or `make serve` for the API.

**No `make`?** (Windows without Git-for-Windows' make, or a bare container.)
Every target is a thin wrapper over the CLI — run it directly:

```bash
uv run python scripts/fetch_data.py --dataset all
uv run fraudsight features && uv run fraudsight train && uv run fraudsight thresholds
uv run fraudsight profiles && uv run fraudsight cases && uv run fraudsight evalset
uv run fraudsight health          # what is present on disk and what is missing
uv run fraudsight eval --repeat 3 --floor
uv run fraudsight gate --min-cases 100 --require-repeats 3
uv run streamlit run src/fraudsight/console/app.py   # the console
uv run fraudsight serve                              # the API
```

`fraudsight health` is the fastest way to see what state a checkout is in — it
lists every artifact and the command that produces it.

### ⚠ Do not put `.venv` inside OneDrive, Dropbox or iCloud

A synced folder holds file handles open while it uploads. Package installs into a
venv under one fail **mid-transaction** — the package manager removes the old
files, cannot write the new ones, and leaves a half-installed package behind.
`pyarrow` is the usual casualty, and because pandas imports it at module level
the breakage cascades to pandas → scikit-learn → LightGBM, so the failure
surfaces as `AttributeError: module 'pyarrow' has no attribute '__version__'`
somewhere unrelated to what you were installing.

Symptom of a half-install:

```bash
uv pip install --python .venv/Scripts/python.exe --reinstall pyarrow   # repairs it
```

The fix is to keep the environment off the synced drive:

```bash
uv venv --python 3.12 C:/venvs/fraudsight     # or ~/.venvs/fraudsight
uv pip install --python C:/venvs/fraudsight/Scripts/python.exe -e ".[all]"
```

The repository itself is fine in a synced folder — only the virtualenv is a
problem, because it is the only thing being rewritten in place.

### What you get without an API key

The pipeline runs end to end and produces valid briefs, but they come from the
**rule-based provider**, not a language model. That is a real shipped component
([ADR 0007](docs/adr/0007-deterministic-provider-as-floor.md)) and it is what
makes the CI gate runnable — but it is the *floor* the LLM is measured against,
so with no key every agent-vs-floor delta in the results table is 0.0000 and the
LLM contributes nothing because it is never called.

To actually exercise the agent:

```bash
cp .env.example .env      # set ANTHROPIC_API_KEY and PRIMARY_PROVIDER=anthropic
uv run fraudsight health  # confirms which providers are credentialed
uv run fraudsight eval --repeat 3 --floor
```

Everything runs against a **deterministic rule-based provider** by default. That
is not a stub — it implements the same `LLMClient` protocol, is scored by the
same eval suite, and is the last rung of the fallback ladder in production
([ADR 0007](docs/adr/0007-deterministic-provider-as-floor.md)). To use a real
model:

```bash
cp .env.example .env      # set ANTHROPIC_API_KEY, PRIMARY_PROVIDER=anthropic
make eval                 # same suite, now with a real delta over the floor
```

| Command | What it does |
|---|---|
| `make eval` | Triage eval suite, 3 repeats, agent vs floor |
| `make gate` | Fails the build if quality dropped. Exits non-zero. |
| `make redteam` | 40 injection attempts through the live pipeline |
| `make replay` | Stream historical transactions at a target rate |
| `make report` | Daily self-report: funnel, quality, cost, drift |
| `make test` | 113 unit + security tests, no services needed |
| `make up` | Postgres+pgvector, Redis, Langfuse, Grafana (all optional) |

---

## How it works

```
transaction
   ↓
features (velocity windows on event time, 186 model features)
   ↓
LightGBM + isotonic calibration           0.78 ms p99
   ↓
├── score < 0.054  → auto-approve         86.3% of volume, $0
├── score ≥ 0.531  → auto-block            1.3%
└── otherwise      → triage               12.4%
                        ↓
     redact → gather → retrieve → reason → validate → (repair)*
       PII    3 tools   hybrid    LLM      policy     bounded
              parallel  search
                        ↓
              TriageBrief (schema-validated)
                        ↓
     ├── confidence ≥ 0.80 + clean → act
     └── otherwise                 → analyst console → decision captured
                                                          ↓
                                            eval set + retraining data
```

### Eight decisions worth defending

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-gbdt-in-hot-path.md) | A GBDT, not an LLM, scores every transaction |
| [0002](docs/adr/0002-llm-only-for-triage.md) | The LLM writes briefs; it does not decide |
| [0003](docs/adr/0003-pgvector-over-pinecone.md) | Local index → pgvector; no dedicated vector DB |
| [0004](docs/adr/0004-no-agent-framework.md) | Hand-written state machine; LangGraph behind an adapter |
| [0005](docs/adr/0005-expected-cost-thresholds.md) | Bands by expected cost — and the recall NFR is missed |
| [0006](docs/adr/0006-deterministic-redaction-before-presidio.md) | Deterministic redaction is the control; Presidio is depth |
| [0007](docs/adr/0007-deterministic-provider-as-floor.md) | A rule-based provider ships as fallback, floor and CI path |
| [0008](docs/adr/0008-retry-and-fallback-policy.md) | Retry only what is retryable; never fail over on a budget breach |

Also worth reading: **[docs/scope.md](docs/scope.md)** — everything deliberately
not built, with the measurement behind each decision. It is the more useful half
of the design.

And two postmortems on this project's own worst bugs:

- **[the hallucination metric that was measuring its own regex](docs/postmortems/2026-08-15-groundedness-false-positives.md)**
  — four classes of false positive, and the rule that came out of it: if the
  system expects a brief to cite a number, that number must be *in* the evidence.
- **[16 tps, and the model was not the problem](docs/postmortems/2026-08-15-feature-store-throughput.md)**
  — an O(n²) sliding window and a sentinel treated as an entity. 30× throughput,
  and a fabricated velocity signal removed.

---

## Layout

```
src/fraudsight/
  config.py          pydantic-settings, SecretStr keys, the cost matrix
  features/          spec (one shared implementation), offline, online, store
  scoring/           train, calibration, thresholds, serve (the fast path)
  llm/               base protocol, anthropic, openai, deterministic, router, usage
  triage/            evidence, signals, prompts/, graph, rules, pipeline
  tools/             4 read-only scoped tools + hybrid vector store
  guardrails/        pii, injection, output policy
  evals/             judges, calibration, metrics, runner, gate, redteam
  observability/     tracing (JSONL/Langfuse/OTel), cost + daily report
  api/ console/ replay/
tests/               unit (leak, parity, policy, router) · security · integration
docs/adr/            8 architecture decision records
evals/datasets/      the 150-case eval set, in git
```

---

## Data

**IEEE-CIS Fraud Detection** — 590,540 transactions, real labels. Split
chronologically with a 14-day chargeback embargo (never randomly — fraud
drifts, and a random split lets the model see the future). PaySim is fetched for
stream-shape replay.

`scripts/fetch_data.py` pulls both without Kaggle credentials.

---

## Honest summary

**What this demonstrates:** architectural restraint under a latency and cost
budget; expected-cost threshold selection with a published frontier; a leak-safe
time-split with an embargo; a schema-validated agent with a bounded repair loop;
groundedness enforced mechanically; a real CI gate that runs without a vendor
key; and a postmortem about the project's own worst bug.

**What it does not:** meet the recall NFR (0.736 vs 0.95, priced and explained);
hit 2,000 tps in one process (~600–750, needs 3–4 workers); validate the LLM
judge against human labels; or measure the LLM's contribution over the
rule-based floor, because this run had no API key.

Each of those is a line in [docs/scope.md](docs/scope.md) rather than an
omission you had to find.
