"""Small YAML configuration loader with no network or interpolation side effects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_config(path: Path) -> dict[str, Any]:
    """Load a mapping configuration from a local YAML file."""

    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    return values
