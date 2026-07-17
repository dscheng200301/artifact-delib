"""Modality-isolated Text and Image Agent adapters."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.schemas import Label, ModelRequest, TokenUsage


@dataclass(frozen=True)
class AgentEvidence:
    label: Label | None
    evidence: dict[str, Any]
    modality: str
    usage: TokenUsage


class TextAgent:
    """Send only caption text to the configured LLM client."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def analyze(self, caption: str) -> AgentEvidence:
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt="You are Text Agent. Analyze only the supplied caption claim.",
                user_prompt=caption,
            )
        )
        return AgentEvidence(
            _read_label(response.content), {"raw": response.content}, "text", response.usage
        )


class ImageAgent:
    """Send only a local image payload to the configured VLM client."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def analyze(self, image_path: Path) -> AgentEvidence:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt="You are Image Agent. Analyze only visible image evidence.",
                user_prompt="Analyze the supplied image without reading a caption.",
                image_base64=f"data:image/png;base64,{encoded}",
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
