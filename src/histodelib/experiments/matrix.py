"""Expand YAML experiment matrices into explicit, non-executing plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from histodelib.methods.baselines import BASELINE_NAMES


@dataclass(frozen=True)
class ExperimentPlan:
    matrix_name: str
    method: str
    config: str
    formal_dataset: str


def plan_experiments(path: Path) -> list[ExperimentPlan]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("experiment matrix root must be a mapping")
    methods = values.get("methods", [])
    configs = values.get("configs", [])
    dataset = str(values.get("formal_dataset", "NOT_SELECTED"))
    if not isinstance(methods, list) or not isinstance(configs, list) or not methods or not configs:
        raise ValueError("experiment matrix requires non-empty methods and configs lists")
    if dataset not in {"NOT_SELECTED", "SYNTHETIC_FIXTURE"}:
        raise PermissionError("formal dataset execution requires explicit authorization")
    unknown = sorted(set(str(method) for method in methods) - set(BASELINE_NAMES))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    return [
        ExperimentPlan(str(values.get("name", path.stem)), str(method), str(config), dataset)
        for method in methods
        for config in configs
    ]
