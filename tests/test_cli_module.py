from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_module_entrypoint_invokes_typer_app() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    result = subprocess.run(
        [sys.executable, "-m", "histodelib.cli", "doctor"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "HistoDelib doctor" in result.stdout


def test_fixture_run_writes_guarded_cache_state(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histodelib.cli",
            "run",
            "--method",
            "direct_vlm",
            "--config",
            "guarded-test",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_path = tmp_path / "outputs" / "direct_vlm-guarded-test-synthetic" / "call_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records
    assert records[0]["cache_state"] == "miss"


def test_cli_run_loads_yaml_config_into_artifact(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histodelib.cli",
            "run",
            "--method",
            "direct_vlm",
            "--config",
            str(ROOT / "configs" / "run" / "fixture.yaml"),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dirs = list((tmp_path / "outputs").glob("direct_vlm-*-synthetic"))
    assert len(run_dirs) == 1
    config_text = (run_dirs[0] / "config.resolved.yaml").read_text(encoding="utf-8")
    assert "name: fixture" in config_text
    assert "synthetic_only: true" in config_text


def test_cli_config_model_reaches_call_log(tmp_path: Path) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text(
        "name: custom-model-run\nmode: fixture\nsynthetic_only: true\n"
        "model: custom-model\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histodelib.cli",
            "run",
            "--method",
            "direct_vlm",
            "--config",
            str(config),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_path = tmp_path / "outputs" / "direct_vlm-custom-model-run-synthetic" / "call_log.jsonl"
    assert '"model": "custom-model"' in log_path.read_text(encoding="utf-8")


def test_api_mode_requires_explicit_paid_call_and_configuration(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "API_ALLOW_PAID_CALLS": "false",
        "LLM_API_KEY": "",
        "VLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "VLM_BASE_URL": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histodelib.cli",
            "run",
            "--mode",
            "api",
            "--method",
            "direct_vlm",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "API_ALLOW_PAID_CALLS" in result.stderr or "API_ALLOW_PAID_CALLS" in result.stdout


def test_method_yaml_parameters_drive_fixture_execution(tmp_path: Path) -> None:
    config = tmp_path / "api-deliberation.yaml"
    config.write_text(
        "name: api-deliberation\nmode: fixture\nsynthetic_only: true\n"
        "enable_api_deliberation: true\nmax_reinspection_targets: 1\n"
        "max_cross_exam_rounds: 1\nmodel: custom-model\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "API_MAX_TOTAL_REQUESTS": "100",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "histodelib.cli",
            "run",
            "--method",
            "histodelib_rule",
            "--config",
            str(config),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    prediction = json.loads(
        (tmp_path / "outputs" / "histodelib_rule-api-deliberation-synthetic" / "predictions.jsonl")
        .read_text(encoding="utf-8").splitlines()[4]
    )
    assert prediction["evidence"]["judge_api_calls"] == 1
    assert prediction["evidence"]["reinspection_api_calls"] <= 1
