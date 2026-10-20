.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := uv run

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install
install:  ## Install the package and dev extras
	uv sync --all-extras

.PHONY: data
data:  ## Fetch IEEE-CIS and PaySim (no credentials needed)
	$(PY) python scripts/fetch_data.py --dataset all

# ---------------------------------------------------------------------------
# Pipeline. Each target depends on the artifact the previous one writes, so
# `make artifacts` rebuilds only what is missing.
# ---------------------------------------------------------------------------

data/processed/train.parquet:
	$(PY) fraudsight features

artifacts/model.txt: data/processed/train.parquet
	$(PY) fraudsight train

artifacts/thresholds.json: artifacts/model.txt
	$(PY) fraudsight thresholds

artifacts/profiles/customers.json: data/processed/train.parquet
	$(PY) fraudsight profiles

artifacts/cases/cases.jsonl: artifacts/profiles/customers.json
	$(PY) fraudsight cases

evals/datasets/triage_holdout.jsonl: artifacts/thresholds.json artifacts/cases/cases.jsonl
	$(PY) fraudsight evalset

.PHONY: features train thresholds profiles cases evalset artifacts
features: data/processed/train.parquet            ## Build leak-safe splits
train: artifacts/model.txt                        ## Train + calibrate the scorer
thresholds: artifacts/thresholds.json             ## Pick bands by expected cost
profiles: artifacts/profiles/customers.json       ## Derive entity profiles
cases: artifacts/cases/cases.jsonl                ## Build the retrieval corpus
evalset: evals/datasets/triage_holdout.jsonl      ## Sample the 150-case eval set
artifacts: evalset                                ## Everything the agent needs

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: eval
eval: artifacts  ## Run the triage eval suite (3 repeats, with the rule-based floor)
	$(PY) fraudsight eval --repeat 3 --floor

.PHONY: calibrate
calibrate: artifacts  ## Measure the LLM judge against the labelled set
	$(PY) fraudsight calibrate --build-reference

.PHONY: gate
gate:  ## Fail the build if quality dropped. Exits non-zero.
	$(PY) fraudsight gate --min-cases 100 --require-repeats 3

.PHONY: redteam
redteam: artifacts  ## Run the prompt-injection suite
	$(PY) fraudsight redteam

.PHONY: test
test:  ## Unit + security tests (no services, no API key)
	$(PY) pytest tests/unit tests/security -q

.PHONY: test-all
test-all:  ## Everything, including integration
	$(PY) pytest -q

.PHONY: lint
lint:  ## Ruff + mypy
	$(PY) ruff check src tests scripts
	$(PY) ruff format --check src tests scripts
	$(PY) mypy src

.PHONY: fix
fix:  ## Auto-fix what ruff can
	$(PY) ruff check --fix src tests scripts
	$(PY) ruff format src tests scripts

.PHONY: check
check: lint test gate  ## What CI runs

# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

.PHONY: serve
serve: artifacts  ## FastAPI on :8000
	$(PY) fraudsight serve

.PHONY: console
console: artifacts  ## Streamlit analyst console on :8501
	$(PY) streamlit run src/fraudsight/console/app.py

.PHONY: replay
replay: artifacts  ## Replay 5,000 transactions through the live path
	$(PY) fraudsight replay --n 5000 --triage-sample 0.15

.PHONY: throughput
throughput: artifacts  ## Scoring-path throughput, no LLM calls
	$(PY) fraudsight replay --n 50000 --triage-sample 0.0

.PHONY: report
report:  ## Daily self-report: funnel, quality, cost, drift
	$(PY) fraudsight report

.PHONY: health
health:  ## What is present on disk and what is missing
	$(PY) fraudsight health

.PHONY: demo
demo: artifacts  ## Everything a reviewer needs, from zero, no API key
	@echo "=== 1/4  system state ==="
	@$(PY) fraudsight health
	@echo ""
	@echo "=== 2/4  one transaction, end to end ==="
	@$(PY) fraudsight triage $$(head -1 evals/datasets/triage_holdout.jsonl | $(PY) python -c "import sys,json; print(json.load(sys.stdin)['case_id'])")
	@echo ""
	@echo "=== 3/4  replay 2,000 transactions ==="
	@$(PY) fraudsight replay --n 2000 --triage-sample 0.1
	@echo ""
	@echo "=== 4/4  daily report ==="
	@$(PY) fraudsight report
	@echo ""
	@echo "Now run 'make console' for the analyst UI."

# ---------------------------------------------------------------------------
# Infrastructure (optional -- everything above runs without it)
# ---------------------------------------------------------------------------

.PHONY: up
up:  ## Postgres+pgvector, Redis, Langfuse, Grafana
	docker compose up -d

.PHONY: down
down:
	docker compose down -v

.PHONY: clean
clean:  ## Remove derived artifacts (keeps raw data)
	rm -rf artifacts/traces artifacts/reports artifacts/*.json artifacts/*.npy artifacts/model.txt
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
