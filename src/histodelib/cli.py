"""Typer commands for safe local fixture checks and mock-only runs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import typer

from histodelib.api.budget import BudgetManager
from histodelib.api.cache import ResponseCache
from histodelib.api.call_log import CallLogStore
from histodelib.api.factory import build_remote_client, selected_model
from histodelib.api.guarded import GuardedModelClient
from histodelib.api.mock import MockModelClient
from histodelib.config import load_config
from histodelib.config_schema import validate_runtime_config
from histodelib.data.fixture_builder import build_fixture
from histodelib.data.importer import import_manifest
from histodelib.data.validator import validate_samples
from histodelib.experiments.matrix import plan_experiments
from histodelib.methods.baselines import BASELINE_NAMES, create_baseline
from histodelib.reporting import write_run_report
from histodelib.runner.run_manager import RunManager
from histodelib.settings import Settings
from histodelib.validation.smoke import validate_smoke_artifacts

app = typer.Typer(help="HistoDelib API-only engineering commands.", no_args_is_help=True)
fixture_app = typer.Typer(help="Build and validate synthetic test fixtures.")
data_app = typer.Typer(help="Validate local dataset manifests without downloading data.")
report_app = typer.Typer(help="Render reports from structured run artifacts.")
app.add_typer(fixture_app, name="fixture")
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")


@app.command()
def doctor() -> None:
    """Print safe runtime checks without contacting any API."""

    settings = Settings()
    typer.echo("HistoDelib doctor")
    typer.echo(f"run mode: {settings.histodelib_run_mode}")
    typer.echo(f"paid calls enabled: {settings.api_allow_paid_calls}")
    typer.echo("formal dataset: NOT_SELECTED")
    typer.echo("formal experiment: NOT_RUN")


@fixture_app.command("build")
def fixture_build(root: Path = typer.Option(Path("data/fixtures"))) -> None:
    """Generate deterministic images labelled solely for fixture verification."""

    samples = build_fixture(root)
    typer.echo(f"built {len(samples)} samples at {root}")
    typer.echo("SYNTHETIC_FIXTURE")
    typer.echo("NOT_FOR_RESEARCH_RESULTS")


@fixture_app.command("validate")
def fixture_validate(root: Path = typer.Option(Path("data/fixtures"))) -> None:
    """Rebuild deterministic fixture definitions and validate their files."""

    report = validate_samples(build_fixture(root))
    if not report.is_valid:
        for error in report.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("fixture valid: SYNTHETIC_FIXTURE; NOT_FOR_RESEARCH_RESULTS")


@data_app.command("import")
def data_import(manifest: Path = typer.Option(...), image_root: Path = typer.Option(...)) -> None:
    """Import and validate a user-provided local manifest."""

    samples = import_manifest(manifest, image_root)
    report = validate_samples(samples)
    if not report.is_valid:
        for error in report.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"imported {len(samples)} samples")
    typer.echo("formal dataset remains NOT_SELECTED until explicitly authorized")


@report_app.command("run")
def report_run(run_id: str, output_root: Path = typer.Option(Path("outputs"))) -> None:
    """Render a report from an existing structured run directory."""

    run_dir = output_root / run_id
    if not (run_dir / "predictions.jsonl").exists():
        raise typer.BadParameter(f"run artifacts not found: {run_dir}")
    report_path = write_run_report(run_dir)
    typer.echo(f"report written: {report_path}")


@app.command()
def run(
    method: str = typer.Option("histodelib_rule"),
    config: str = typer.Option("fixture"),
    mode: str = typer.Option("fixture", help="fixture or api"),
    output_root: Path = typer.Option(Path("outputs")),
    allow_paid_calls: bool = typer.Option(False),
) -> None:
    """Run synthetic fixtures through mock or explicitly authorized remote API clients."""

    if mode not in {"fixture", "api"}:
        raise typer.BadParameter("mode must be fixture or api")
    if method not in BASELINE_NAMES:
        raise typer.BadParameter(f"unsupported fixture method: {method}")
    samples = build_fixture(Path("data/fixtures"))
    config_path = Path(config)
    run_config = (
        load_config(config_path)
        if config_path.exists()
        else {"name": config, "mode": "fixture", "synthetic_only": True}
    )
    method_config_path = Path("configs/method") / f"{method}.yaml"
    method_config = load_config(method_config_path) if method_config_path.exists() else {}
    resolved_config = validate_runtime_config({**method_config, **run_config})
    config_name = str(resolved_config.get("name", config_path.stem or config))
    safe_config_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config_name).strip("-") or "fixture"
    settings = Settings()
    model_name = str(resolved_config.get("model") or selected_model(settings))
    if mode == "fixture":
        if allow_paid_calls:
            raise typer.BadParameter("use --mode api for remote calls")
        run_id = f"{method}-{safe_config_name}-synthetic"
        guarded_client = GuardedModelClient(
            MockModelClient(role="vlm"),
            cache=ResponseCache(output_root / run_id / "cache"),
            budget=BudgetManager(
                max_requests=settings.api_max_total_requests,
                max_tokens=settings.api_max_total_tokens,
                max_cost=settings.api_max_estimated_cost,
            ),
            call_log=CallLogStore(output_root / run_id / "call_log.jsonl"),
            max_retries=settings.api_max_retries,
        )
    else:
        if allow_paid_calls:
            settings = settings.model_copy(update={"api_allow_paid_calls": True})
        if not settings.api_allow_paid_calls:
            raise typer.BadParameter("set API_ALLOW_PAID_CALLS=true before an API run")
        run_id = f"{method}-{safe_config_name}-api-synthetic"
        guarded_client = build_remote_client(settings, output_root / run_id)
        resolved_config = {
            **validate_runtime_config(resolved_config),
            "mode": "api",
            "synthetic_only": True,
            "enable_api_deliberation": True,
        }
    baseline = create_baseline(
        method,
        guarded_client,
        model_name=model_name,
        enable_api_deliberation=bool(resolved_config.get("enable_api_deliberation", False)),
        max_reinspection_targets=int(resolved_config.get("max_reinspection_targets", 2)),
        max_cross_exam_rounds=int(resolved_config.get("max_cross_exam_rounds", 2)),
    )
    summary = RunManager(output_root).run(
        samples,
        baseline,
        run_id=run_id,
        resolved_config=resolved_config,
        mode=mode,
    )
    run_dir = summary.run_dir
    (run_dir / "README.txt").write_text(
        "SYNTHETIC_FIXTURE\nNOT_FOR_RESEARCH_RESULTS\n",
        encoding="utf-8",
    )
    run_kind = "remote API synthetic smoke" if mode == "api" else "mock fixture"
    typer.echo(f"{run_kind} run complete: {run_dir}")
    typer.echo("SYNTHETIC_FIXTURE; NOT_FOR_RESEARCH_RESULTS")


@app.command("clean-generated")
def clean_generated() -> None:
    """Remove only reproducible local outputs, never source data or credentials."""

    output_root = Path("outputs")
    if output_root.exists():
        shutil.rmtree(output_root)
    typer.echo("generated outputs removed")


@app.command("validate-smoke")
def validate_smoke(
    run_dir: Path = typer.Argument(...),
    expected_predictions: int = typer.Option(12, min=1),
) -> None:
    """Fail closed when a synthetic API smoke artifact is incomplete."""

    result = validate_smoke_artifacts(run_dir, expected_predictions=expected_predictions)
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"smoke valid: predictions={result.prediction_count}; "
        f"api_calls={result.api_calls}; providers={','.join(result.providers)}"
    )


@app.command("experiment-plan")
def experiment_plan(matrix: Path = typer.Argument(...)) -> None:
    """Print a dry-run experiment matrix; this command never calls an API."""

    plans = plan_experiments(matrix)
    for plan in plans:
        typer.echo(f"{plan.matrix_name}\t{plan.method}\t{plan.config}\t{plan.formal_dataset}")


if __name__ == "__main__":
    app()
