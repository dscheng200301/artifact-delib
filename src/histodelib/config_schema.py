"""Validation for runtime YAML values while preserving provenance fields."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "fixture"
    mode: Literal["fixture", "api"] = "fixture"
    synthetic_only: bool = True
    model: str | None = None
    router: Literal["rule", "api", "always_full"] | None = None
    enable_api_deliberation: bool = False
    max_reinspection_targets: int = Field(default=2, ge=0, le=16)
    max_cross_exam_rounds: int = Field(default=2, ge=0, le=16)


def validate_runtime_config(values: dict[str, Any]) -> dict[str, Any]:
    """Validate known runtime controls and retain unknown provenance fields."""

    try:
        return RuntimeConfig.model_validate(values).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ValueError(f"invalid runtime configuration: {exc}") from exc
