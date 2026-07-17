"""Typer commands for safe local fixture checks and mock-only runs."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from histodelib.api.mock import MockModelClient
from histodelib.data.fixture_builder import build_fixture
from histodelib.data.importer import import_manifest
from histodelib.data.validator import validate_samples
from histodelib.methods.baselines import BASELINE_NAMES, create_baseline
from histodelib.reporting import write_run_report
from histodelib.runner.run_manager import RunManager
from histodelib.settings import Settings

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
    output_root: Path = typer.Option(Path("outputs")),
    allow_paid_calls: bool = typer.Option(False),
) -> None:
    """Run the deterministic mock path; remote calls require future explicit configuration."""

    if allow_paid_calls:
        raise typer.BadParameter(
            "real API smoke runs are intentionally not implemented in fixture mode"
        )
    if method not in BASELINE_NAMES:
        raise typer.BadParameter(f"unsupported fixture method: {method}")
    samples = build_fixture(Path("data/fixtures"))
    baseline = create_baseline(method, MockModelClient(role="vlm"))
    run_id = f"{method}-{config}-synthetic"
    summary = RunManager(output_root).run(samples, baseline, run_id=run_id)
    run_dir = summary.run_dir
    (run_dir / "README.txt").write_text(
        "SYNTHETIC_FIXTURE\nNOT_FOR_RESEARCH_RESULTS\n",
        encoding="utf-8",
    )
    typer.echo(f"mock fixture run complete: {run_dir}")
    typer.echo("SYNTHETIC_FIXTURE; NOT_FOR_RESEARCH_RESULTS")


@app.command("clean-generated")
def clean_generated() -> None:
    """Remove only reproducible local outputs, never source data or credentials."""

    output_root = Path("outputs")
    if output_root.exists():
        shutil.rmtree(output_root)
    typer.echo("generated outputs removed")


if __name__ == "__main__":
    app()
