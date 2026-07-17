"""Short relation probe used for routing, not final adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelationProbeResult:
    risk_flags: tuple[str, ...]
    summary: str


class LightRelationProbe:
    """Detect structured disagreement without deciding the gold label."""

    def assess(
        self, text_evidence: dict[str, Any], image_evidence: dict[str, Any]
    ) -> RelationProbeResult:
        flags: list[str] = []
        if text_evidence.get("label") != image_evidence.get("label"):
            flags.append("modality_disagreement")
        if text_evidence.get("claims") and not image_evidence.get("visible_text"):
            flags.append("unreadable_glyph")
        return RelationProbeResult(tuple(flags), ";".join(flags) or "no immediate risk")
