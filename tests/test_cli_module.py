from __future__ import annotations

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
