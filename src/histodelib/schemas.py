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

    @computed_field(return_type=int)
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
    split: Literal["train", "validation", "test"] | None = None
    fixture_markers: set[str] = Field(default_factory=set)

    @field_validator("caption")
    @classmethod
    def caption_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("caption must not be blank")
        return value

    @computed_field(return_type=bool)
    @property
    def is_synthetic_fixture(self) -> bool:
        return {"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"}.issubset(
            self.fixture_markers
        )


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
