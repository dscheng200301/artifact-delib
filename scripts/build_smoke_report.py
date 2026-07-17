"""Build a synthetic-only smoke report from structured prediction artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/histodelib_rule-fixture-synthetic"))
    parser.add_argument("--output", type=Path, default=Path("reports/smoke_test_report.md"))
    args = parser.parse_args()
    predictions_path = args.run_dir / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in rows)
    total_calls = sum(row.get("api_calls", 0) for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Mock Smoke Test Report",
                "",
                "SYNTHETIC_FIXTURE",
                "NOT_FOR_RESEARCH_RESULTS",
                "",
                f"Run directory: `{args.run_dir}`",
                f"Samples: {len(rows)}",
                f"Total tokens: {total_tokens}",
                f"Total API calls: {total_calls}",
                "Real API Smoke Test: NOT_RUN",
                "Formal Dataset: NOT_SELECTED",
                "Formal Experiment: NOT_RUN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
