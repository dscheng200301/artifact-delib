"""Typed data exchanged between ArtifactDelib components.

Extracted from the model client infrastructure:
- TokenUsage, ModelRequest, ModelResponse
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenUsage(BaseModel):
    """Provider-reported or locally estimated token use for one request."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @computed_field(return_type=int)  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
