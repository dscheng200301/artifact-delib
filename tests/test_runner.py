from __future__ import annotations

import json
from pathlib import Path

from histodelib.data.fixture_builder import build_fixture
from histodelib.runner.run_manager import RunManager
from histodelib.schemas import Prediction, TokenUsage


class CountingMethod:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, sample):
        self.calls += 1
        return Prediction(
            sample_id=sample.sample_id,
            method="counting",
            final_label=sample.label,
            initial_label=sample.label,
            usage=TokenUsage(input_tokens=2, output_tokens=1),
            api_calls=1,
        )


def test_run_manager_writes_structured_artifacts_and_resumes(tmp_path: Path) -> None:
    samples = build_fixture(tmp_path / "fixtures")
    method = CountingMethod()
    manager = RunManager(tmp_path / "outputs")

    first = manager.run(samples, method, run_id="fixture-run")
    second = manager.run(samples, method, run_id="fixture-run")
    run_dir = tmp_path / "outputs" / "fixture-run"

    assert first.completed_samples == 3
    assert second.completed_samples == 3
    assert method.calls == 3
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert (
        json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["mode"] == "fixture"
    )
