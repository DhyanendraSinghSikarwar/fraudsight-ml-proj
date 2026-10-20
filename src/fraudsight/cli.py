"""`fraudsight` command line.

One entry point per pipeline stage, in dependency order:

    data -> features -> train -> thresholds -> profiles -> cases -> evalset
         -> eval -> calibrate -> gate -> replay -> report

Each command writes an artifact and prints the numbers it produced. Commands are
independently runnable against artifacts on disk, which is what lets CI run
`eval` and `gate` without retraining, and what lets a reviewer run `demo`
without downloading 1.2GB of raw CSV.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from fraudsight.config import settings

app = typer.Typer(
    add_completion=False,
    help="Real-time fraud triage: GBDT in the hot path, LLM in the investigation layer.",
    no_args_is_help=True,
)
console = Console()


def _ok(msg: str) -> None:
    console.print(f"[green]OK[/green] {msg}")


def _warn(msg: str) -> None:
    console.print(f"[yellow]WARN[/yellow] {msg}")


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------


@app.command()
def features(
    nrows: int | None = typer.Option(None, help="Limit rows, for a fast smoke run."),
    embargo_days: int = typer.Option(14, help="Chargeback maturity embargo after the train window."),
) -> None:
    """Build leak-safe train/val/test splits from the raw IEEE-CIS tables."""
    from fraudsight.features.offline import build

    summary = build(nrows=nrows, embargo_days=embargo_days)
    console.print_json(json.dumps(summary))
    _ok(f"{summary['rows_total']:,} rows, {summary['features']} features")


@app.command()
def train(
    seeds: str = typer.Option("13,29,41", help="Comma-separated seeds; mean +- std is reported."),
) -> None:
    """Train LightGBM + isotonic calibration across seeds."""
    from fraudsight.scoring.train import train as run_train

    report = run_train(seeds=tuple(int(s) for s in seeds.split(",")))
    t = Table(title="Scorer — held-out test")
    t.add_column("metric")
    t.add_column("mean", justify="right")
    t.add_column("std", justify="right")
    for k, v in report["test_metrics"].items():
        t.add_row(k, f"{v['mean']:.4f}", f"{v['std']:.4f}")
    console.print(t)
    _ok(f"artifacts/model.txt (seed {report['chosen_seed']}, {report['selection_rule']})")


@app.command()
def thresholds() -> None:
    """Pick score bands by minimising expected cost, and publish the frontier."""
    import numpy as np

    from fraudsight.scoring import thresholds as th
    from fraudsight.scoring.dataset import artifact

    costs = th.CostMatrix(
        settings.fraud_loss_usd,
        settings.false_decline_usd,
        settings.analyst_review_usd,
        settings.triage_llm_usd,
    )
    vp, vy = np.load(artifact("val_probs.npy")), np.load(artifact("val_labels.npy"))
    tp, ty = np.load(artifact("test_probs.npy")), np.load(artifact("test_labels.npy"))

    report = th.select_operating_point(
        vp, vy, costs,
        min_recall=settings.band_min_recall,
        max_review_rate=settings.band_max_review_rate,
    )
    lo, hi = report["chosen"]["lo"], report["chosen"]["hi"]
    held = th.evaluate_bands(tp, ty, lo, hi, costs)
    report["held_out_test"] = held.__dict__
    th.save(report, artifact("thresholds.json"))

    t = Table(title="Operating point (chosen on val, reported on test)")
    for c in ("", "val", "test"):
        t.add_column(c or "metric", justify="right" if c else "left")
    chosen = report["chosen"]
    for key in ("recall", "review_rate", "auto_approve_rate", "auto_block_rate", "cost_per_txn_usd"):
        t.add_row(key, f"{chosen[key]:.4f}", f"{getattr(held, key):.4f}")
    console.print(t)

    f = Table(title="Recall / cost frontier — what each recall floor would cost")
    for c in ("recall target", "achieved", "review rate", "$/txn"):
        f.add_column(c, justify="right")
    for row in report["frontier"]:
        if row.get("feasible"):
            f.add_row(
                f"{row['recall_target']:.2f}",
                f"{row['achieved_recall']:.3f}",
                f"{row['review_rate']:.3f}",
                f"${row['cost_per_txn_usd']:.3f}",
            )
    console.print(f)

    if not report["nfr_recall_met"]:
        _warn(
            f"recall NFR of {settings.band_min_recall:.2f} not met at review rate "
            f"<= {settings.band_max_review_rate:.2f}. Shipped the cost-optimal point "
            f"(recall {chosen['recall']:.3f}). See docs/scope.md."
        )
    _ok(f"bands lo={lo:.4f} hi={hi:.4f} -> artifacts/thresholds.json")


@app.command()
def profiles() -> None:
    """Derive customer / device / merchant profiles from the training window."""
    from fraudsight.data import profiles as p

    tables = p.build()
    tables.save(settings.artifact_dir / "profiles")
    console.print_json(json.dumps(tables.built_from))
    _ok("artifacts/profiles/")


@app.command()
def cases(size: int = typer.Option(1500, help="Corpus size.")) -> None:
    """Build and embed the resolved-case retrieval corpus."""
    from fraudsight.data.cases import build_corpus
    from fraudsight.data.profiles import ProfileTables

    store = build_corpus(ProfileTables.load(settings.artifact_dir / "profiles"), size=size)
    fraud = sum(1 for c in store.cases if c.outcome == "confirmed_fraud")
    _ok(f"{len(store)} cases ({fraud} fraud) -> artifacts/cases/")


@app.command()
def evalset(per_stratum: int = typer.Option(50)) -> None:
    """Sample the labelled triage eval set from the val window."""
    from fraudsight.data.cases import build_eval_cases, save_eval_cases
    from fraudsight.scoring import thresholds as th

    lo, hi = th.load_bands(settings.artifact_dir / "thresholds.json")
    rows = build_eval_cases(lo=lo, hi=hi, per_stratum=per_stratum)
    path = save_eval_cases(rows)
    counts: dict[str, int] = {}
    for c in rows:
        counts[c.stratum] = counts.get(c.stratum, 0) + 1
    console.print_json(json.dumps(counts))
    _ok(f"{len(rows)} cases -> {path}")


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


@app.command()
def triage(
    transaction_id: str = typer.Argument(..., help="A case_id or transaction id from the eval set."),
) -> None:
    """Run one transaction through the full pipeline and print the brief."""
    from fraudsight.data.cases import load_eval_cases
    from fraudsight.triage.pipeline import TriagePipeline

    cases_ = load_eval_cases()
    match = next(
        (c for c in cases_ if transaction_id in (c.case_id, c.transaction_id)), None
    )
    if match is None:
        raise typer.BadParameter(f"{transaction_id} not in the eval set")

    async def go() -> Any:
        pipe = TriagePipeline.build()
        return await pipe.process(match.transaction)

    decision = asyncio.run(go())
    console.print_json(json.dumps(decision.as_dict(), default=str))


@app.command()
def eval(
    repeat: int = typer.Option(1, help="Runs of the whole suite; mean +- std reported above 1."),
    limit: int | None = typer.Option(None, help="Truncate the case list, for a fast check."),
    judge: bool = typer.Option(True, help="Run the LLM judges."),
    floor: bool = typer.Option(True, help="Also run the rule-based floor and report the delta."),
    out: Path | None = typer.Option(None, help="Report path."),
) -> None:
    """Run the triage eval suite and write the report."""
    from fraudsight.evals import runner

    cfg = runner.RunConfig(repeat=repeat, judge=judge, limit=limit)
    coro = runner.run_with_floor(run_cfg=cfg) if floor else runner.run(run_cfg=cfg)
    report = asyncio.run(coro)
    path = runner.save(report, out)

    m = report["metrics"]
    t = Table(title="Triage agent — Track B")
    for c in ("metric", "agent", "floor", "delta"):
        t.add_column(c, justify="left" if c == "metric" else "right")
    fl = (report.get("floor") or {}).get("metrics", {})
    vs = report.get("vs_floor") or {}

    def row(label: str, a: Any, f: Any, d: Any) -> None:
        fmt = lambda v: "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        t.add_row(label, fmt(a), fmt(f), fmt(d))

    row("agreement (macro)", m.get("agreement_macro"), fl.get("agreement_macro"),
        vs.get("agreement_macro_delta"))
    row("decisive rate", m.get("decisive_rate"), fl.get("decisive_rate"),
        vs.get("decisive_rate_delta"))
    row("schema pass", m["schema"]["pass_rate"], (fl.get("schema") or {}).get("pass_rate"), None)
    row("numeric grounding", m["groundedness"]["numeric_claim_grounding"],
        (fl.get("groundedness") or {}).get("numeric_claim_grounding"), None)
    row("narrative score", m["narrative"]["mean_score"],
        (fl.get("narrative") or {}).get("mean_score"), vs.get("narrative_score_delta"))
    row("$ / 1k triages", m["cost"]["per_1000_triages_usd"],
        (fl.get("cost") or {}).get("per_1000_triages_usd"), vs.get("cost_delta_per_1000_usd"))
    console.print(t)

    s = Table(title="Agreement by stratum")
    for c in ("stratum", "n", "agent", "floor", "delta"):
        s.add_column(c, justify="left" if c == "stratum" else "right")
    for name, v in (vs.get("per_stratum") or {}).items():
        s.add_row(
            name,
            str(m["per_stratum"][name]["n"]),
            f"{v['agent']:.4f}" if v["agent"] is not None else "n/a",
            f"{v['floor']:.4f}" if v["floor"] is not None else "n/a",
            f"{v['delta']:+.4f}" if v["delta"] is not None else "n/a",
        )
    console.print(s)
    _ok(f"report -> {path}")


@app.command()
def calibrate(
    build_reference: bool = typer.Option(
        False, "--build-reference", help="Regenerate the rules-based reference labels."
    ),
    n: int = typer.Option(30, help="Calibration set size."),
) -> None:
    """Measure the LLM judge against the labelled set and report Cohen's kappa."""
    from fraudsight.data.cases import load_eval_cases
    from fraudsight.evals import calibrate as cal
    from fraudsight.evals.judges import Judge
    from fraudsight.llm import build_router
    from fraudsight.triage.pipeline import TriagePipeline

    async def go() -> dict[str, Any]:
        pipe = TriagePipeline.build(write_traces=False)
        rows = load_eval_cases()
        by_stratum: dict[str, list[Any]] = {}
        for c in rows:
            by_stratum.setdefault(c.stratum, []).append(c)
        picked = [c for group in by_stratum.values() for c in group[: max(1, n // 3)]]

        pairs = []
        for case in picked:
            d = await pipe.process(case.transaction)
            if d.state and d.state.brief:
                pairs.append((d.state.brief, d.state.bundle))

        if build_reference:
            p = cal.build_reference_labels(pairs, limit=n)
            _warn(f"wrote rules-based reference labels -> {p} (NOT human labels)")

        return await cal.calibrate_judge(Judge(build_router()), pairs)

    report = asyncio.run(go())
    path = cal.save(report)

    t = Table(title=f"Judge calibration — {report['comparison']}")
    for c in ("metric", "kappa", "95% CI", "agreement", "n", "verdict"):
        t.add_column(c)
    for r in report["reports"]:
        t.add_row(
            r["metric"],
            f"{r['cohens_kappa']:.3f}",
            f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]",
            f"{r['raw_agreement']:.3f}",
            str(r["n_human_labels"]),
            r["interpretation"],
        )
    console.print(t)
    _warn(report["caveat"])
    _ok(f"-> {path}")


@app.command()
def label(
    n: int = typer.Option(30, help="How many cases to label."),
    out: Path = typer.Option(
        Path("evals/datasets/human_labels.jsonl"), help="Where to write the labels."
    ),
) -> None:
    """Interactively label briefs, producing the judge calibration set.

    Follow evals/labelling_protocol.md. The confirmed outcome is deliberately
    NOT shown: you are judging the brief against its evidence, not against
    reality, and knowing the answer is the bias the judge is being tested for.
    """
    from fraudsight.data.cases import load_eval_cases
    from fraudsight.triage.pipeline import TriagePipeline

    rows = load_eval_cases()
    by_stratum: dict[str, list[Any]] = {}
    for c in rows:
        by_stratum.setdefault(c.stratum, []).append(c)
    per = max(1, n // max(len(by_stratum), 1))
    picked = [c for group in by_stratum.values() for c in group[:per]][:n]

    async def brief_for(case: Any) -> Any:
        pipe = _label_pipeline()
        return await pipe.process(case.transaction)

    labels: list[dict[str, Any]] = []
    out_path = out if out.is_absolute() else (settings.artifact_dir.parent / out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(picked, 1):
        decision = asyncio.run(brief_for(case))
        state = decision.state
        if state is None or state.brief is None:
            continue
        brief = state.brief

        console.rule(f"[bold]{i}/{len(picked)}  {case.case_id}")
        console.print("[bold]Evidence bundle[/bold]")
        console.print_json(state.bundle.to_json())
        console.print("\n[bold]Brief[/bold]")
        console.print_json(json.dumps(brief.model_dump(mode="json"), default=str))

        grounded = typer.prompt(
            "\nGrounded? every claim present in the evidence (1 = yes, 0 = no)", type=int
        )
        score = typer.prompt("Narrative quality 1-5 (rubric in judge_narrative.v1.md)", type=int)
        labels.append(
            {
                "transaction_id": brief.transaction_id,
                "grounded": 1 if grounded else 0,
                "narrative_score": max(1, min(5, score)),
                "labeller": "human",
                "rubric_version": "v1",
            }
        )
        with out_path.open("w", encoding="utf-8") as fh:
            for row in labels:
                fh.write(json.dumps(row) + "\n")

    _ok(f"{len(labels)} human labels -> {out_path}")
    console.print("Now run `fraudsight calibrate` — it prefers human labels automatically.")


def _label_pipeline() -> Any:
    from fraudsight.triage.pipeline import TriagePipeline

    if not hasattr(_label_pipeline, "_cached"):
        _label_pipeline._cached = TriagePipeline.build(write_traces=False)  # type: ignore[attr-defined]
    return _label_pipeline._cached  # type: ignore[attr-defined]


@app.command()
def gate(
    report: Path | None = typer.Option(None, help="Eval report path."),
    min_cases: int = typer.Option(100),
    require_repeats: int = typer.Option(1),
) -> None:
    """Fail the build if quality dropped below the floors. Exits non-zero."""
    from fraudsight.evals.gate import load_and_evaluate

    result = load_and_evaluate(report, min_cases=min_cases, require_repeats=require_repeats)
    console.print(result.render())
    raise typer.Exit(0 if result.passed else 1)


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


@app.command()
def replay(
    n: int = typer.Option(2000, help="Transactions to replay."),
    rate: float = typer.Option(0.0, help="Target transactions/sec; 0 = as fast as possible."),
    split: str = typer.Option("test", help="Which split to stream."),
    triage_sample: float = typer.Option(
        1.0, help="Fraction of triage-band transactions to actually triage."
    ),
) -> None:
    """Stream historical transactions through the live path at a target rate."""
    from fraudsight.replay.harness import ReplayHarness

    async def go() -> dict[str, Any]:
        harness = await ReplayHarness.build(split=split, triage_sample=triage_sample)
        return await harness.run(n=n, target_rate=rate)

    summary = asyncio.run(go())
    console.print_json(json.dumps(summary, default=str))

    # The daily report cannot see the funnel from traces alone -- only triage
    # runs are traced. Persisting the scoring path's band counts is what lets
    # `fraudsight report` show the real split instead of guessing.
    out = settings.artifact_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _ok(f"replayed {summary['transactions']:,} at {summary['achieved_tps']:.0f} tps")


@app.command()
def report(
    day: str | None = typer.Option(None, help="YYYY-MM-DD; defaults to today (UTC)."),
    all_traces: bool = typer.Option(
        False, "--all", help="Aggregate the whole trace history instead of one day."
    ),
    transactions: int | None = typer.Option(
        None, help="Total transactions scored, for the funnel denominator."
    ),
) -> None:
    """Daily self-report: funnel, quality, cost, drift (capability C8)."""
    from datetime import UTC, datetime

    import numpy as np

    from fraudsight.observability import cost as cost_mod
    from fraudsight.observability.tracing import read_traces
    from fraudsight.scoring.dataset import artifact

    target = day or datetime.now(UTC).strftime("%Y-%m-%d")
    traces = read_traces(day=None if all_traces else target)
    if not traces:
        _warn(
            f"no traces for {target} — run `fraudsight replay` first, or pass --all "
            f"to aggregate every run"
        )
        raise typer.Exit(1)

    # The funnel comes from the last replay's scoring-path counts when present.
    band_counts: dict[str, int] | None = None
    last = settings.artifact_dir / "reports" / "last_replay.json"
    if last.exists():
        summary = json.loads(last.read_text(encoding="utf-8"))
        band_counts = summary.get("band_counts")
        transactions = transactions or summary.get("transactions")

    ref = artifact("val_probs.npy")
    reference = np.load(ref) if ref.exists() else None
    daily = cost_mod.build(
        traces,
        reference_scores=reference,
        day=target,
        total_transactions=transactions,
        band_counts=band_counts,
    )
    console.print(daily.render())
    _ok(f"-> {cost_mod.save(daily)}")


@app.command()
def redteam(
    out: Path | None = typer.Option(None, help="Where to write the red-team report."),
) -> None:
    """Run the prompt-injection suite and publish the pass rate."""
    from fraudsight.evals.redteam import run_suite

    result = asyncio.run(run_suite())
    path = out or (settings.artifact_dir / "redteam_report.json")
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    t = Table(title="Red team — prompt injection")
    for c in ("category", "attempts", "blocked", "rate"):
        t.add_column(c, justify="right" if c != "category" else "left")
    for cat, v in result["by_category"].items():
        t.add_row(cat, str(v["attempts"]), str(v["blocked"]), f"{v['rate']:.0%}")
    console.print(t)
    console.print(
        f"\n[bold]{result['blocked']}/{result['attempts']} attacks blocked "
        f"({result['pass_rate']:.0%})[/bold]"
    )
    for gap in result["known_gaps"]:
        _warn(gap)
    _ok(f"-> {path}")


@app.command()
def health() -> None:
    """What is present on disk and what is missing."""
    from fraudsight.scoring.dataset import processed_exists

    checks = [
        ("raw IEEE-CIS", (settings.data_dir / "raw" / "ieee_train_transaction.csv").exists(), "fraudsight-data"),
        ("processed splits", processed_exists(), "fraudsight features"),
        ("scorer", (settings.artifact_dir / "model.txt").exists(), "fraudsight train"),
        ("calibrator", (settings.artifact_dir / "calibrator.json").exists(), "fraudsight train"),
        ("thresholds", (settings.artifact_dir / "thresholds.json").exists(), "fraudsight thresholds"),
        ("profiles", (settings.artifact_dir / "profiles" / "customers.json").exists(), "fraudsight profiles"),
        ("case corpus", (settings.artifact_dir / "cases" / "cases.jsonl").exists(), "fraudsight cases"),
        ("eval set", (settings.artifact_dir.parent / "evals" / "datasets" / "triage_holdout.jsonl").exists(), "fraudsight evalset"),
        ("eval report", (settings.artifact_dir / "eval_report.json").exists(), "fraudsight eval"),
    ]
    t = Table(title="fraudsight health")
    for c in ("artifact", "present", "produced by"):
        t.add_column(c)
    for name, present, cmd in checks:
        t.add_row(name, "[green]yes[/green]" if present else "[red]no[/red]", cmd)
    console.print(t)

    t2 = Table(title="providers")
    for c in ("setting", "value"):
        t2.add_column(c)
    t2.add_row("primary", settings.primary_provider)
    t2.add_row("fallback", str(settings.fallback_provider))
    t2.add_row("available", ", ".join(settings.available_providers()))
    t2.add_row("triage model", settings.triage_model)
    t2.add_row("vector backend", settings.vector_backend)
    console.print(t2)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Run the FastAPI service (/score, /triage, /decision, /health)."""
    import uvicorn

    uvicorn.run("fraudsight.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
