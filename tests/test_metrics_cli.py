from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from histodelib.cli import app
from histodelib.metrics import compute_metrics
from histodelib.schemas import Label, Prediction


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


def test_fixture_cli_builds_and_validates_synthetic_data(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["fixture", "build", "--root", str(tmp_path)])
    validation = runner.invoke(app, ["fixture", "validate", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "SYNTHETIC_FIXTURE" in result.output
    assert validation.exit_code == 0, validation.output
    assert "valid" in validation.output.lower()
