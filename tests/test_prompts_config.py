from __future__ import annotations

from pathlib import Path

from histodelib.config import load_config
from histodelib.prompts.loader import load_prompt


def test_prompt_loader_renders_and_hashes_versioned_yaml(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        "name: demo\nversion: v1\nrole: text\n"
        "max_output_tokens: 20\ntemperature: 0\n"
        "system_prompt: 'Role: {{ role }}'\n"
        "user_template: 'Claim: {{ claim }}'\n",
        encoding="utf-8",
    )

    prompt = load_prompt(prompt_path)
    rendered = prompt.render({"role": "Text Agent", "claim": "1912"})

    assert prompt.version == "v1"
    assert len(prompt.content_hash) == 64
    assert rendered.user_prompt == "Claim: 1912"


def test_config_loader_reads_yaml_without_external_services(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: fixture\napi:\n  allow_paid_calls: false\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["mode"] == "fixture"
    assert config["api"]["allow_paid_calls"] is False


def test_repository_has_required_prompt_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("text_agent", "image_agent", "relation_probe", "cross_exam", "judge"):
        assert (root / "prompts" / name / "v1.yaml").exists()


def test_repository_defaults_to_qwen35_flash_for_all_modalities() -> None:
    root = Path(__file__).resolve().parents[1]
    assert "qwen3.5-flash-2026-02-23" in (root / "configs" / "api" / "llm.yaml").read_text()
    assert "qwen3.5-flash-2026-02-23" in (root / "configs" / "api" / "vlm.yaml").read_text()
