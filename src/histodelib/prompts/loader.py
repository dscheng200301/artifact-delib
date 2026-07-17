"""Load, hash, and render versioned YAML prompts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jinja2 import Template


@dataclass(frozen=True)
class RenderedPrompt:
    system_prompt: str
    user_prompt: str
    version: str
    content_hash: str
    max_output_tokens: int
    temperature: float


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    role: str
    system_prompt: str
    user_template: str
    max_output_tokens: int
    temperature: float
    content_hash: str

    def render(self, context: dict[str, Any]) -> RenderedPrompt:
        return RenderedPrompt(
            system_prompt=Template(self.system_prompt).render(**context),
            user_prompt=Template(self.user_template).render(**context),
            version=self.version,
            content_hash=self.content_hash,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )


def load_prompt(path: Path) -> PromptSpec:
    raw = path.read_bytes()
    values = yaml.safe_load(raw) or {}
    required = {"name", "version", "role", "system_prompt", "user_template"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"prompt missing fields: {sorted(missing)}")
    return PromptSpec(
        name=str(values["name"]),
        version=str(values["version"]),
        role=str(values["role"]),
        system_prompt=str(values["system_prompt"]),
        user_template=str(values["user_template"]),
        max_output_tokens=int(values.get("max_output_tokens", 512)),
        temperature=float(values.get("temperature", 0.0)),
        content_hash=hashlib.sha256(raw).hexdigest(),
    )
