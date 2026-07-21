"""Pydantic configuration models for ExperimentConfig.

All experiment configuration is validated through these models,
which map directly to the YAML config files in configs/experiments/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════
#  DatasetConfig
# ═══════════════════════════════════════════════════════════════


class DatasetConfig(BaseModel):
    """Dataset configuration for an experiment run."""

    manifest: str
    image_root: str
    split_file: str | None = None
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = 42

    @field_validator("manifest")
    @classmethod
    def manifest_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("manifest path must not be empty")
        return v


# ═══════════════════════════════════════════════════════════════
#  ExecutionConfig
# ═══════════════════════════════════════════════════════════════


class ExecutionConfig(BaseModel):
    """Execution behavior for the experiment."""

    allow_remote_calls: bool = False
    allow_model_download: bool = False
    max_retries: int = Field(default=3, ge=0)
    retry_delay_s: float = Field(default=5.0, ge=0)
    concurrency: int = Field(default=1, ge=1)


# ═══════════════════════════════════════════════════════════════
#  ModelConfig
# ═══════════════════════════════════════════════════════════════


class ModelConfig(BaseModel):
    """Model selection for the experiment."""

    name: str = Field(min_length=1)


# ═══════════════════════════════════════════════════════════════
#  OutputConfig
# ═══════════════════════════════════════════════════════════════


class OutputConfig(BaseModel):
    """Output directory and save options."""

    dir: str = "results"
    save_raw_responses: bool = True
    save_predictions: bool = True
    save_metrics: bool = True
    save_failures: bool = True


# ═══════════════════════════════════════════════════════════════
#  AccountingConfig
# ═══════════════════════════════════════════════════════════════


class AccountingConfig(BaseModel):
    """Accounting configuration."""

    track_tokens: bool = True
    track_latency: bool = True
    track_cost: bool = True
    distinguish_cache_hits: bool = True


# ═══════════════════════════════════════════════════════════════
#  LeakageConfig
# ═══════════════════════════════════════════════════════════════


class LeakageConfig(BaseModel):
    """Leakage detection configuration."""

    run_before_experiment: bool = True
    fail_on_exact_duplicate: bool = True
    fail_on_object_overlap: bool = True
    warn_on_near_duplicate: bool = True
    warn_on_filename_leakage: bool = True


# ═══════════════════════════════════════════════════════════════
#  SuccessCriteria
# ═══════════════════════════════════════════════════════════════


class SuccessCriteria(BaseModel):
    """Pilot / smoke success criteria."""

    min_completed_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    max_parse_failure_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    require_all_route_types: bool = False
    validate_token_accounting: bool = True


# ═══════════════════════════════════════════════════════════════
#  ExperimentConfig
# ═══════════════════════════════════════════════════════════════


class ExperimentInfo(BaseModel):
    """Experiment metadata."""

    name: str = Field(min_length=1)
    description: str = ""


class ExperimentConfig(BaseModel):
    """Top-level validated experiment configuration.

    Maps directly to the YAML structure in configs/experiments/*.yaml.
    """

    experiment: ExperimentInfo
    dataset: DatasetConfig
    methods: list[str] = Field(min_length=1)
    model: ModelConfig
    execution: ExecutionConfig = ExecutionConfig()
    output: OutputConfig = OutputConfig()
    accounting: AccountingConfig = AccountingConfig()
    leakage_detection: LeakageConfig = LeakageConfig()
    pilot_success_criteria: SuccessCriteria = SuccessCriteria()

    @field_validator("methods")
    @classmethod
    def methods_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one method must be specified")
        for m in v:
            if not m.strip():
                raise ValueError(f"Invalid method name: {m!r}")
        return v

    @model_validator(mode="after")
    def _check_remote_execution(self) -> ExperimentConfig:
        if self.execution.allow_remote_calls and not self.execution.allow_model_download:
            # Remote calls without model download is fine (API-only methods)
            pass
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file."""
        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error in {path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")

        return cls(**data)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Serialize config back to a plain dict (for snapshot)."""
        return self.model_dump(mode="python")


# ═══════════════════════════════════════════════════════════════
#  Pricing registry
# ═══════════════════════════════════════════════════════════════


class PricingEntry(BaseModel):
    """Pricing for one model."""

    model: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    currency: str = "USD"
    source: str = "estimated"
    effective_date: str | None = None


# Default pricing — only models we know actual prices for.
# For unknown models, estimated_cost will be null.
_DEFAULT_PRICING: dict[str, PricingEntry] = {
    "qwen3.5-flash-2026-02-23": PricingEntry(
        model="qwen3.5-flash-2026-02-23",
        input_price_per_million=None,
        output_price_per_million=None,
        source="unknown",
        effective_date=None,
    ),
}


class PricingRegistry:
    """Central pricing registry for all models.

    Prices are looked up by model name. If the price is unknown,
    estimated_cost will be None — never fabricate prices.
    """

    def __init__(self, entries: dict[str, PricingEntry] | None = None) -> None:
        self._entries: dict[str, PricingEntry] = dict(_DEFAULT_PRICING)
        if entries:
            self._entries.update(entries)

    def get(self, model: str) -> PricingEntry | None:
        """Get pricing entry for a model, or None if unknown."""
        return self._entries.get(model)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float | None:
        """Estimate cost for a call, or None if pricing is unknown."""
        entry = self._entries.get(model)
        if entry is None:
            return None
        if entry.input_price_per_million is None or entry.output_price_per_million is None:
            return None
        cost = (
            input_tokens * entry.input_price_per_million
            + output_tokens * entry.output_price_per_million
        ) / 1_000_000
        return cost

    def register(
        self,
        model: str,
        input_price: float | None,
        output_price: float | None,
        source: str = "estimated",
        currency: str = "USD",
        effective_date: str | None = None,
    ) -> None:
        """Register or update pricing for a model."""
        self._entries[model] = PricingEntry(
            model=model,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            currency=currency,
            source=source,
            effective_date=effective_date,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> PricingRegistry:
        """Load pricing from a YAML file."""
        import yaml

        if not path.exists():
            return cls()

        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        entries = {}
        for entry_data in data.get("pricing", []):
            entry = PricingEntry(**entry_data)
            entries[entry.model] = entry
        return cls(entries)


# Convenience singleton
GLOBAL_PRICING = PricingRegistry()