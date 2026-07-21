"""Optional local cost estimation from user-maintained pricing configuration."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_pricing(path: str | Path, model: str) -> dict[str, Any]:
    """Load one model's local pricing entry, preserving the currency metadata."""

    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = values.get("models", {}) if isinstance(values, dict) else {}
    entry = models.get(model, {}) if isinstance(models, dict) else {}
    if not isinstance(entry, dict):
        entry = {}
    return {"currency": values.get("currency", "USD"), **entry}


def estimate_cost(
    input_tokens: int, output_tokens: int, pricing: dict[str, object] | None
) -> float | None:
    """Estimate USD cost only when both per-million-token prices are supplied."""

    if not pricing:
        return None
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    if not isinstance(input_price, (int, float)) or not isinstance(output_price, (int, float)):
        return None
    amount = (
        Decimal(input_tokens) * Decimal(str(input_price))
        + Decimal(output_tokens) * Decimal(str(output_price))
    ) / Decimal(1_000_000)
    return float(amount)
