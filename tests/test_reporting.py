from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from histodelib.cli import app
from histodelib.reporting import write_run_report
from histodelib.schemas import Label, Prediction, TokenUsage


def test_run_report_reads_structured_predictions_and_marks_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction = Prediction(
        sample_id="a",
        method="mock",
        final_label=Label.TRUE,
        usage=TokenUsage(input_tokens=2, output_tokens=3),
        api_calls=1,
    )
    (run_dir / "predictions.jsonl").write_text(
        prediction.model_dump_json() + "\n", encoding="utf-8"
    )

    report_path = write_run_report(run_dir)

    text = report_path.read_text(encoding="utf-8")
    assert "SYNTHETIC_FIXTURE" in text
    assert "Total tokens: 5" in text


def test_report_cli_materializes_report_for_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run-1"
    run_dir.mkdir(parents=True)
    prediction = Prediction(sample_id="a", method="mock", final_label=Label.TRUE)
    (run_dir / "predictions.jsonl").write_text(
        prediction.model_dump_json() + "\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        app, ["report", "run", "run-1", "--output-root", str(tmp_path / "outputs")]
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "report.md").exists()
