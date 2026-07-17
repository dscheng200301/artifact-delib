from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_api_only_python_312_runtime() -> None:
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    base_requirements = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    api_requirements = (ROOT / "requirements" / "api.txt").read_text(encoding="utf-8")

    assert "python=3.12" in environment
    declared = f"{base_requirements}\n{api_requirements}".lower()
    for forbidden in ("torch", "transformers", "ollama", "vllm", "open-clip"):
        assert forbidden not in declared


def test_gitignore_protects_env_credentials() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored.splitlines()
