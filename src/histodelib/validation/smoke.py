"""Fail-closed validation for synthetic API smoke artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from histodelib.schemas import Prediction


@dataclass(frozen=True)
class SmokeValidation:
    ok: bool
    errors: tuple[str, ...]
    prediction_count: int
    completed_count: int
    api_calls: int
    error_count: int
    providers: tuple[str, ...]


def validate_smoke_artifacts(
    run_dir: Path,
    *,
    expected_predictions: int | None = None,
    expected_mode: str = "api",
) -> SmokeValidation:
    errors: list[str] = []
    prediction_path = run_dir / "predictions.jsonl"
    call_log_path = run_dir / "call_log.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    predictions: list[Prediction] = []
    if not prediction_path.exists():
        errors.append("predictions.jsonl is missing")
    else:
        for line_no, line in enumerate(prediction_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                predictions.append(Prediction.model_validate_json(line))
            except ValueError as exc:
                errors.append(f"prediction line {line_no} is invalid: {exc}")
    if expected_predictions is not None and len(predictions) != expected_predictions:
        errors.append(f"expected {expected_predictions} predictions, found {len(predictions)}")
    completed = [prediction for prediction in predictions if prediction.status == "COMPLETED"]
    if len(completed) != len(predictions):
        errors.append("all predictions must have status COMPLETED")
    if any(prediction.final_label is None for prediction in predictions):
        errors.append("all predictions must have a final_label")

    records: list[dict[str, object]] = []
    if not call_log_path.exists():
        errors.append("call_log.jsonl is missing")
    else:
        for line_no, line in enumerate(call_log_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"call log line {line_no} is invalid JSON")
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                errors.append(f"call log line {line_no} is not an object")
    error_count = sum(1 for record in records if record.get("error_type"))
    if error_count:
        errors.append(f"call log contains {error_count} error records")
    providers = tuple(
        sorted({str(record["provider"]) for record in records if record.get("provider")})
    )
    if not providers:
        errors.append("call log contains no provider records")

    if not metadata_path.exists():
        errors.append("run_metadata.json is missing")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("mode") != expected_mode:
                errors.append(f"run metadata mode must be {expected_mode}")
        except json.JSONDecodeError:
            errors.append("run_metadata.json is invalid JSON")
    return SmokeValidation(
        ok=not errors,
        errors=tuple(errors),
        prediction_count=len(predictions),
        completed_count=len(completed),
        api_calls=sum(prediction.api_calls for prediction in predictions),
        error_count=error_count,
        providers=providers,
    )
