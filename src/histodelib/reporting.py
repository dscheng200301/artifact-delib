"""Human-readable reports derived only from structured run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from histodelib.schemas import Prediction


def read_predictions(run_dir: Path) -> list[Prediction]:
    """Load and validate every prediction row in a run directory."""

    path = run_dir / "predictions.jsonl"
    return [
        Prediction.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_run_report(run_dir: Path) -> Path:
    """Write a fixture-safe Markdown summary and return its path."""

    predictions = read_predictions(run_dir)
    total_tokens = sum(prediction.usage.total_tokens for prediction in predictions)
    total_calls = sum(prediction.api_calls for prediction in predictions)
    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# HistoDelib Run Report",
                "",
                "SYNTHETIC_FIXTURE",
                "NOT_FOR_RESEARCH_RESULTS",
                "",
                f"Samples: {len(predictions)}",
                f"Total tokens: {total_tokens}",
                f"Total API calls: {total_calls}",
                "Formal experiment: NOT_RUN",
                "Formal dataset: NOT_SELECTED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
