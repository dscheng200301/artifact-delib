from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from histodelib.cli import app
from histodelib.data.fixture_builder import build_fixture
from histodelib.metrics import compute_metrics
from histodelib.schemas import Label, Prediction, TokenUsage


def test_metrics_are_calculated_from_structured_predictions_only() -> None:
    predictions = [
        Prediction(sample_id="a", method="mock", final_label=Label.TRUE),
        Prediction(sample_id="b", method="mock", final_label=Label.MISCAPTIONED),
        Prediction(sample_id="c", method="mock", final_label=Label.OUT_OF_CONTEXT),
    ]
    labels = {"a": Label.TRUE, "b": Label.MISCAPTIONED, "c": Label.OUT_OF_CONTEXT}

    metrics = compute_metrics(predictions, labels)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.token_saving is None


def test_metrics_include_token_saving_correction_and_harm() -> None:
    predictions = [
        Prediction(
            sample_id="a",
            method="adaptive",
            initial_label=Label.MISCAPTIONED,
            final_label=Label.TRUE,
            usage=TokenUsage(input_tokens=5, output_tokens=5),
        ),
        Prediction(
            sample_id="b",
            method="adaptive",
            initial_label=Label.TRUE,
            final_label=Label.MISCAPTIONED,
            usage=TokenUsage(input_tokens=5, output_tokens=5),
        ),
    ]
    baseline = [
        Prediction(
            sample_id="a",
            method="full",
            final_label=Label.TRUE,
            usage=TokenUsage(input_tokens=20, output_tokens=0),
        ),
        Prediction(
            sample_id="b",
            method="full",
            final_label=Label.TRUE,
            usage=TokenUsage(input_tokens=20, output_tokens=0),
        ),
    ]

    metrics = compute_metrics(
        predictions,
        {"a": Label.TRUE, "b": Label.TRUE},
        baseline_predictions=baseline,
    )

    assert metrics.average_tokens == 10
    assert metrics.token_saving == 0.5
    assert metrics.correction_rate == 1.0
    assert metrics.harm_rate == 1.0


def test_fixture_cli_builds_and_validates_synthetic_data(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["fixture", "build", "--root", str(tmp_path)])
    validation = runner.invoke(app, ["fixture", "validate", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "SYNTHETIC_FIXTURE" in result.output
    assert validation.exit_code == 0, validation.output
    assert "valid" in validation.output.lower()


def test_data_import_cli_validates_local_manifest(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,caption,label\n"
        f"{sample.sample_id},{sample.image_path.name},caption,TRUE\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "data",
            "import",
            "--manifest",
            str(manifest),
            "--image-root",
            str(tmp_path / "images"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "imported 1 samples" in result.output


def test_run_cli_writes_resumable_metadata_and_predictions(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--method",
            "direct_vlm",
            "--config",
            "fixture",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
    )

    run_dir = tmp_path / "outputs" / "direct_vlm-fixture-synthetic"
    assert result.exit_code == 0, result.output
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
