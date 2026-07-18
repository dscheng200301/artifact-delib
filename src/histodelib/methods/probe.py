"""Short relation probe used for routing, not final adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelationProbeResult:
    risk_flags: tuple[str, ...]
    summary: str
    region_candidates: tuple[tuple[float, float, float, float], ...] = ()


class LightRelationProbe:
    """Detect structured disagreement without deciding the gold label."""

    def assess(
        self, text_evidence: dict[str, Any], image_evidence: dict[str, Any]
    ) -> RelationProbeResult:
        flags: list[str] = []
        if text_evidence.get("label") != image_evidence.get("label"):
            flags.append("modality_disagreement")
        if text_evidence.get("requires_visible_text") and not image_evidence.get("visible_text"):
            flags.append("unreadable_glyph")
        for pair in text_evidence.get("claim_fact_pairs", ()):
            if not isinstance(pair, dict):
                continue
            relation = str(pair.get("relation", ""))
            conflict_type = str(pair.get("conflict_type", ""))
            if relation == "contradicted" and conflict_type in {
                "temporal",
                "temporal_conflict",
            }:
                flags.append("temporal_conflict")
            if relation == "contradicted" and conflict_type in {
                "location",
                "location_conflict",
            }:
                flags.append("location_conflict")
            if relation == "contradicted" and conflict_type in {
                "identity",
                "event",
                "identity_conflict",
                "event_conflict",
            }:
                flags.append("identity_conflict")
        flags = list(dict.fromkeys(flags))
        regions = image_evidence.get("region_candidates", ())
        normalized_regions = tuple(
            (
                float(region[0]),
                float(region[1]),
                float(region[2]),
                float(region[3]),
            )
            for region in regions
            if isinstance(region, (list, tuple)) and len(region) == 4
        )
        return RelationProbeResult(
            tuple(flags), ";".join(flags) or "no immediate risk", normalized_regions
        )
