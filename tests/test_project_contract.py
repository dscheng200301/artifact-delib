from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_api_only_python_312_runtime() -> None:
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    base_requirements = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    api_requirements = (ROOT / "requirements" / "api.txt").read_text(encoding="utf-8")

    assert "name: histo-delib" in environment
    assert "python=3.12" in environment
    declared = f"{base_requirements}\n{api_requirements}".lower()
    for forbidden in ("torch", "transformers", "ollama", "vllm", "open-clip"):
        assert forbidden not in declared


def test_runtime_scripts_use_existing_histo_delib_environment() -> None:
    scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "scripts").glob("*") if path.is_file()
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "-n histo-delib" in scripts
    assert "-n histo-delib" in makefile
    assert "conda env create" not in scripts
    assert "conda env create" not in makefile


def test_gitignore_protects_env_credentials() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored.splitlines()
