"""Targeted reinspection selection."""

from __future__ import annotations

from dataclasses import dataclass

from histodelib.methods.probe import RelationProbeResult


@dataclass(frozen=True)
class ReinspectionDecision:
    targets: tuple[str, ...]


class TargetedReinspection:
    """Map probe flags to bounded, interpretable reinspection views."""

    _TARGETS = {
        "modality_disagreement": "patch",
        "unreadable_glyph": "glyph",
        "temporal_conflict": "panor",
        "location_conflict": "panor",
    }

    def select(self, probe: RelationProbeResult) -> ReinspectionDecision:
        targets = tuple(
            dict.fromkeys(self._TARGETS[flag] for flag in probe.risk_flags if flag in self._TARGETS)
        )
        return ReinspectionDecision(targets)
