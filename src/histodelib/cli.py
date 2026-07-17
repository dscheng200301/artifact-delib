"""Typer commands for safe local fixture checks and mock-only runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from histodelib.api.mock import MockModelClient
from histodelib.data.fixture_builder import build_fixture
from histodelib.data.validator import validate_samples
from histodelib.methods.histodelib import HistoDelibMethod
from histodelib.methods.router import RuleRouter
from histodelib.settings import Settings

app = typer.Typer(help="HistoDelib API-only engineering commands.", no_args_is_help=True)
fixture_app = typer.Typer(help="Build and validate synthetic test fixtures.")
app.add_typer(fixture_app, name="fixture")


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
    if method not in {"histodelib_rule", "histodelib_api_router", "direct_vlm", "always_full"}:
        raise typer.BadParameter(f"unsupported fixture method: {method}")
    samples = build_fixture(Path("data/fixtures"))
    runner = HistoDelibMethod(client=MockModelClient(role="vlm"), router=RuleRouter())
    predictions = [runner.run(sample).model_dump(mode="json") for sample in samples]
    run_dir = output_root / f"{method}-{config}-synthetic"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n",
        encoding="utf-8",
    )
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
