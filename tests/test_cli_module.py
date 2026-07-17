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
