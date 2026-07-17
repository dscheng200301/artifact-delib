"""Runtime settings loaded from environment variables without exposing secrets."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Safe-by-default configuration for local fixture and remote API runs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    histodelib_env: str = "development"
    histodelib_run_mode: str = "fixture"
    histodelib_data_root: Path = Path("data")
    histodelib_output_root: Path = Path("outputs")
    histodelib_cache_root: Path = Path(".cache/histodelib")

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_provider: str = "openai_compatible"
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_provider: str = "openai_compatible"
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    judge_model: str | None = None

    api_timeout_seconds: int = Field(default=120, ge=1)
    api_max_retries: int = Field(default=5, ge=0)
    api_max_concurrency: int = Field(default=2, ge=1)
    api_allow_paid_calls: bool = False
    api_max_total_requests: int = Field(default=20, ge=0)
    api_max_total_tokens: int = Field(default=50_000, ge=0)
    api_max_estimated_cost: float = Field(default=5.0, ge=0.0)
    log_level: str = "INFO"
    store_raw_api_responses: bool = False

    def redacted(self) -> Settings:
        """Return a copy that is safe to serialize or write into artifacts."""

        replacements = {
            field_name: "***REDACTED***"
            for field_name in ("llm_api_key", "vlm_api_key", "judge_api_key")
            if getattr(self, field_name) is not None
        }
        return self.model_copy(update=replacements)
