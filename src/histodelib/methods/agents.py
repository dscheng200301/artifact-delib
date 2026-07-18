"""Modality-isolated Text and Image Agent adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.constants import (
    DEFAULT_MODEL,
    EVIDENCE_JSON_INSTRUCTION,
    JSON_RESPONSE_SCHEMA,
)
from histodelib.prompts.loader import PromptSpec, load_prompt
from histodelib.schemas import (
    ImageEvidence,
    Label,
    ModelRequest,
    PromptProvenance,
    TextEvidence,
    TokenUsage,
)


@dataclass(frozen=True)
class AgentEvidence:
    label: Label | None
    evidence: dict[str, Any]
    modality: str
    usage: TokenUsage
    structured: TextEvidence | ImageEvidence


class TextAgent:
    """Send only caption text to the configured LLM client."""

    def __init__(
        self,
        client: ModelClient,
        prompt: PromptSpec | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.client = client
        self.prompt = prompt or _load_default_prompt("text_agent")
        self.model_name = model_name

    def analyze(self, caption: str) -> AgentEvidence:
        rendered = self.prompt.render({"caption": caption}) if self.prompt else None
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=(
                    f"{rendered.system_prompt} {EVIDENCE_JSON_INSTRUCTION}"
                    if rendered
                    else (
                        "You are Text Agent. Analyze only the supplied caption claim. "
                        f"{EVIDENCE_JSON_INSTRUCTION}"
                    )
                ),
                user_prompt=rendered.user_prompt if rendered else caption,
                temperature=rendered.temperature if rendered else 0.0,
                max_output_tokens=rendered.max_output_tokens if rendered else 512,
                response_schema=dict(JSON_RESPONSE_SCHEMA),
                prompt_name=self.prompt.name if self.prompt else "builtin",
                prompt_version=self.prompt.version if self.prompt else "v1",
                prompt_content_hash=(
                    self.prompt.content_hash
                    if self.prompt
                    else _provenance(None, None).content_hash
                ),
            )
        )
        parsed = _safe_object(response.content)
        label = _read_label(response.content)
        provenance = _provenance(self.prompt, rendered)
        structured = TextEvidence(
            evidence_id=f"text-{uuid.uuid4()}",
            entities=_as_strings(parsed.get("entities")),
            event=_as_optional_string(parsed.get("event")),
            location=_as_optional_string(parsed.get("location")),
            date=_as_optional_string(parsed.get("date")),
            caption_claims=(caption,),
            requires_visible_text=bool(parsed.get("requires_visible_text", False)),
            uncertainty=_as_strings(parsed.get("uncertainty")),
            candidate_label=label,
            prompt=provenance,
        )
        return AgentEvidence(
            label, {"raw": response.content, "structured": structured.model_dump(mode="json")},
            "text", response.usage, structured
        )


class ImageAgent:
    """Send only a local image payload to the configured VLM client."""

    def __init__(
        self,
        client: ModelClient,
        prompt: PromptSpec | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.client = client
        self.prompt = prompt or _load_default_prompt("image_agent")
        self.model_name = model_name

    def analyze(self, image_path: Path) -> AgentEvidence:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        rendered = self.prompt.render({}) if self.prompt else None
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=(
                    f"{rendered.system_prompt} {EVIDENCE_JSON_INSTRUCTION}"
                    if rendered
                    else (
                        "You are Image Agent. Analyze only visible image evidence. "
                        f"{EVIDENCE_JSON_INSTRUCTION}"
                    )
                ),
                user_prompt=(
                    rendered.user_prompt
                    if rendered
                    else "Analyze the supplied image without reading a caption."
                ),
                image_base64=f"data:image/png;base64,{encoded}",
                temperature=rendered.temperature if rendered else 0.0,
                max_output_tokens=rendered.max_output_tokens if rendered else 512,
                response_schema=dict(JSON_RESPONSE_SCHEMA),
                prompt_name=self.prompt.name if self.prompt else "builtin",
                prompt_version=self.prompt.version if self.prompt else "v1",
                prompt_content_hash=(
                    self.prompt.content_hash
                    if self.prompt
                    else _provenance(None, None).content_hash
                ),
            )
        )
        parsed = _safe_object(response.content)
        label = _read_label(response.content)
        provenance = _provenance(self.prompt, rendered)
        structured = ImageEvidence(
            evidence_id=f"image-{uuid.uuid4()}",
            visible_entities=_as_strings(parsed.get("visible_entities")),
            visible_scene=_as_strings(parsed.get("visible_scene")),
            visible_text=_as_optional_string(parsed.get("visible_text")),
            temporal_cues=_as_strings(parsed.get("temporal_cues")),
            location_cues=_as_strings(parsed.get("location_cues")),
            region_candidates=_as_regions(parsed.get("region_candidates")),
            uncertainty=_as_strings(parsed.get("uncertainty")),
            candidate_label=label,
            prompt=provenance,
        )
        return AgentEvidence(
            label, {"raw": response.content, "structured": structured.model_dump(mode="json")},
            "image", response.usage, structured
        )


def _read_label(content: str) -> Label | None:
    try:
        parsed = parse_json_object(content)
        return Label(str(parsed.get("label") or parsed.get("candidate_label")))
    except (TypeError, ValueError):
        return None


def _safe_object(content: str) -> dict[str, Any]:
    try:
        from histodelib.api.response_parser import parse_json_object

        parsed = parse_json_object(content)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _as_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_regions(value: object) -> tuple[tuple[float, float, float, float], ...]:
    regions: list[tuple[float, float, float, float]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for region in value:
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                regions.append(tuple(float(item) for item in region))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return tuple(regions)


def _provenance(prompt: PromptSpec | None, rendered: Any) -> PromptProvenance:
    if prompt is not None:
        return PromptProvenance(
            name=prompt.name,
            version=prompt.version,
            content_hash=prompt.content_hash,
        )
    payload = "builtin-agent-prompt"
    return PromptProvenance(
        name="builtin",
        version="v1",
        content_hash=hashlib.sha256(payload.encode()).hexdigest(),
    )


def _load_default_prompt(name: str) -> PromptSpec | None:
    """Load a repository prompt when running from source; keep installs portable."""

    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / name / "v1.yaml"
    return load_prompt(prompt_path) if prompt_path.exists() else None
