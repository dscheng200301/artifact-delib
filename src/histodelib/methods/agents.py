"""Modality-isolated Text and Image Agent adapters."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.prompts.loader import PromptSpec, load_prompt
from histodelib.schemas import Label, ModelRequest, TokenUsage


@dataclass(frozen=True)
class AgentEvidence:
    label: Label | None
    evidence: dict[str, Any]
    modality: str
    usage: TokenUsage


class TextAgent:
    """Send only caption text to the configured LLM client."""

    def __init__(self, client: ModelClient, prompt: PromptSpec | None = None) -> None:
        self.client = client
        self.prompt = prompt or _load_default_prompt("text_agent")

    def analyze(self, caption: str) -> AgentEvidence:
        rendered = self.prompt.render({"caption": caption}) if self.prompt else None
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt=(
                    rendered.system_prompt
                    if rendered
                    else "You are Text Agent. Analyze only the supplied caption claim."
                ),
                user_prompt=rendered.user_prompt if rendered else caption,
                temperature=rendered.temperature if rendered else 0.0,
                max_output_tokens=rendered.max_output_tokens if rendered else 512,
            )
        )
        return AgentEvidence(
            _read_label(response.content), {"raw": response.content}, "text", response.usage
        )


class ImageAgent:
    """Send only a local image payload to the configured VLM client."""

    def __init__(self, client: ModelClient, prompt: PromptSpec | None = None) -> None:
        self.client = client
        self.prompt = prompt or _load_default_prompt("image_agent")

    def analyze(self, image_path: Path) -> AgentEvidence:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        rendered = self.prompt.render({}) if self.prompt else None
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt=(
                    rendered.system_prompt
                    if rendered
                    else "You are Image Agent. Analyze only visible image evidence."
                ),
                user_prompt=(
                    rendered.user_prompt
                    if rendered
                    else "Analyze the supplied image without reading a caption."
                ),
                image_base64=f"data:image/png;base64,{encoded}",
                temperature=rendered.temperature if rendered else 0.0,
                max_output_tokens=rendered.max_output_tokens if rendered else 512,
            )
        )
        return AgentEvidence(
            _read_label(response.content), {"raw": response.content}, "image", response.usage
        )


def _read_label(content: str) -> Label | None:
    try:
        return Label(str(parse_json_object(content).get("label")))
    except (TypeError, ValueError):
        return None


def _load_default_prompt(name: str) -> PromptSpec | None:
    """Load a repository prompt when running from source; keep installs portable."""

    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / name / "v1.yaml"
    return load_prompt(prompt_path) if prompt_path.exists() else None
