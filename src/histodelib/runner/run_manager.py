"""Resumable local runner that persists structured predictions after each sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml  # type: ignore[import-untyped]

from histodelib.schemas import Prediction, Sample


class Runnable(Protocol):
    def run(self, sample: Sample) -> Prediction:
        """Produce one structured prediction."""


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    completed_samples: int
    skipped_samples: int


class RunManager:
    """Write append-only JSONL artifacts and skip already completed samples."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(
        self,
        samples: list[Sample],
        method: Runnable,
        run_id: str,
        resolved_config: dict[str, object] | None = None,
    ) -> RunSummary:
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "predictions.jsonl"
        existing: dict[str, Prediction] = {}
        if prediction_path.exists():
            for line in prediction_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    prediction = Prediction.model_validate(json.loads(line))
                    existing[prediction.sample_id] = prediction
        with prediction_path.open("a", encoding="utf-8") as handle:
            for sample in samples:
                if sample.sample_id in existing:
                    continue
                prediction = method.run(sample)
                handle.write(prediction.model_dump_json() + "\n")
                existing[sample.sample_id] = prediction
        config = {
            "mode": "fixture",
            "synthetic_only": True,
            "formal_experiment": "NOT_RUN",
            **(resolved_config or {}),
        }
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode": "fixture",
                    "formal_dataset": "NOT_SELECTED",
                    "formal_experiment": "NOT_RUN",
                    "prediction_count": len(existing),
                    "resolved_config": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return RunSummary(run_id, run_dir, len(existing), len(samples) - len(existing))
