"""Experiment logger — tracks experiment runs, configurations, and results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """A single experiment run record."""

    experiment_id: str
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    n_samples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    error: str | None = None
    git_commit: str | None = None


class ExperimentLogger:
    """Persistent experiment logger — writes JSONL records to output directory."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._records: list[ExperimentRecord] = []
        self._current: ExperimentRecord | None = None

    def start_experiment(
        self,
        experiment_id: str,
        method: str,
        config: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Start a new experiment and return its record."""
        self._current = ExperimentRecord(
            experiment_id=experiment_id,
            method=method,
            config=config or {},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._current

    def complete_experiment(
        self,
        metrics: dict[str, Any],
        n_samples: int,
        error: str | None = None,
    ) -> ExperimentRecord:
        """Complete the current experiment and persist."""
        if self._current is None:
            raise RuntimeError("No experiment started")
        self._current.completed_at = datetime.now(timezone.utc).isoformat()
        self._current.n_samples = n_samples
        self._current.metrics = metrics
        self._current.status = "FAILED" if error else "COMPLETED"
        self._current.error = error
        self._persist(self._current)
        self._records.append(self._current)
        return self._current

    def _persist(self, record: ExperimentRecord) -> None:
        """Append experiment record to JSONL file."""
        path = self.output_root / "experiments.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load_history(self) -> list[ExperimentRecord]:
        """Load all past experiment records."""
        path = self.output_root / "experiments.jsonl"
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(ExperimentRecord(**json.loads(line)))
        return records

    def save_predictions(
        self,
        experiment_id: str,
        predictions: list[dict[str, Any]],
    ) -> Path:
        """Save structured predictions for an experiment."""
        pred_dir = self.output_root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"{experiment_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        return path
