"""Explainable routing from short disagreement-probe features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    """A bounded routing decision that never reads reference labels."""

    reinspect: bool
    reinspection_targets: tuple[str, ...]
    reason: str


class RuleRouter:
    """Low-cost deterministic routing for structured probe flags."""

    _TARGETS = {
        "temporal_conflict": "text",
        "location_conflict": "panor",
        "identity_conflict": "patch",
        "unreadable_glyph": "glyph",
        "modality_disagreement": "patch",
    }

    def route(self, probe: dict[str, Any]) -> RouteDecision:
        flags = [str(flag) for flag in probe.get("risk_flags", [])]
        targets = tuple(
            dict.fromkeys(
                self._TARGETS[flag] for flag in flags if flag in self._TARGETS
            )
        )
        return RouteDecision(
            reinspect=bool(targets),
            reinspection_targets=targets,
            reason=";".join(flags) if flags else "low-risk agreement",
        )
