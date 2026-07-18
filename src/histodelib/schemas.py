"""Typed data exchanged between HistoDelib components."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Label(StrEnum):
    """The three task labels used by the verification task."""

    TRUE = "TRUE"
    MISCAPTIONED = "MISCAPTIONED"
    OUT_OF_CONTEXT = "OUT_OF_CONTEXT"


class TokenUsage(BaseModel):
    """Provider-reported or locally estimated token use for one request."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @computed_field(return_type=int)  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PromptProvenance(BaseModel):
    """Immutable identity of the versioned prompt used for evidence extraction."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)


class TextEvidence(BaseModel):
    """Claims extracted from caption text without inspecting the image."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1)
    entities: tuple[str, ...] = ()
    event: str | None = None
    location: str | None = None
    date: str | None = None
    caption_claims: tuple[str, ...] = ()
    requires_visible_text: bool = False
    uncertainty: tuple[str, ...] = ()
    candidate_label: Label | None = None
    prompt: PromptProvenance


class ImageEvidence(BaseModel):
    """Visible facts extracted from an image without inspecting the caption."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1)
    visible_entities: tuple[str, ...] = ()
    visible_scene: tuple[str, ...] = ()
    visible_text: str | None = None
    temporal_cues: tuple[str, ...] = ()
    location_cues: tuple[str, ...] = ()
    region_candidates: tuple[tuple[float, float, float, float], ...] = ()
    uncertainty: tuple[str, ...] = ()
    candidate_label: Label | None = None
    prompt: PromptProvenance


class ClaimFactPair(BaseModel):
    """Explicit alignment between one caption claim and one visual fact."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(min_length=1)
    visual_fact: str | None = None
    relation: Literal["supported", "contradicted", "uncertain", "missing"]
    conflict_type: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = ()


class Sample(BaseModel):
    """A single image-caption verification input."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    image_path: Path
    caption: str = Field(min_length=1)
    label: Label | None = None
    original_group_id: str | None = None
    source: str | None = None
    license: str | None = None
    domain: str | None = None
    conflict_type: str | None = None
    data_version: str | None = None
    annotation_version: str | None = None
    annotator_ids: tuple[str, ...] = ()
    adjudication_status: str | None = None
    caption_source: str | None = None
    label_rationale: str | None = None
    image_sha256: str | None = None
    split: Literal["train", "validation", "test", "fixture", "unassigned"] | None = None
    fixture_markers: set[str] = Field(default_factory=set)

    @field_validator("caption")
    @classmethod
    def caption_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("caption must not be blank")
        return value

    @computed_field(return_type=bool)  # type: ignore[prop-decorator]
    @property
    def is_synthetic_fixture(self) -> bool:
        return {"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"}.issubset(self.fixture_markers)


class ModelRequest(BaseModel):
    """A provider-independent request for an LLM or VLM completion."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    model: str
    system_prompt: str
    user_prompt: str
    image_base64: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1)
    response_schema: dict[str, Any] | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_content_hash: str | None = None


class ModelResponse(BaseModel):
    """A normalized response returned by a configured model client."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    provider: str
    model: str
    raw_response_stored: bool = False


class CallRecord(BaseModel):
    """A redacted audit record for a model call."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0)
    estimated_cost: float | None = Field(default=None, ge=0.0)
    cache_state: Literal["hit", "miss", "disabled"]
    error_type: str | None = None


class Prediction(BaseModel):
    """Structured output used as the sole source for later metrics."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    method: str
    final_label: Label | None
    initial_label: Label | None = None
    status: str = "COMPLETED"
    evidence: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    api_calls: int = Field(default=0, ge=0)
    run_fingerprint: str | None = None
