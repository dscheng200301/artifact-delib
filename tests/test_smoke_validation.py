from __future__ import annotations

import json
from pathlib import Path

from histodelib.schemas import Label, Prediction
from histodelib.validation.smoke import validate_smoke_artifacts


def _write_run(root: Path, *, complete: bool = True) -> None:
    root.mkdir(parents=True)
    prediction = Prediction(
        sample_id="sample-1",
        method="direct_vlm",
        final_label=Label.TRUE if complete else None,
        status="COMPLETED" if complete else "INSUFFICIENT_EVIDENCE",
        api_calls=1,
    )
    (root / "predictions.jsonl").write_text(prediction.model_dump_json() + "\n", encoding="utf-8")
    (root / "call_log.jsonl").write_text(
        json.dumps({"provider": "openai_compatible", "error_type": None}) + "\n",
        encoding="utf-8",
    )
    (root / "run_metadata.json").write_text(json.dumps({"mode": "api"}), encoding="utf-8")


def test_smoke_validation_accepts_completed_api_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    result = validate_smoke_artifacts(run_dir, expected_predictions=1)

    assert result.ok is True
    assert result.errors == ()


def test_smoke_validation_rejects_insufficient_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, complete=False)

    result = validate_smoke_artifacts(run_dir, expected_predictions=1)

    assert result.ok is False
    assert any("COMPLETED" in error for error in result.errors)
