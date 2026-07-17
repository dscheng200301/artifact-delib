"""Explainable routing from short disagreement-probe features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from histodelib.api.base import ModelClient
from histodelib.schemas import ModelRequest


@dataclass(frozen=True)
class RouteDecision:
    """A bounded routing decision that never reads reference labels."""

    reinspect: bool
    reinspection_targets: tuple[str, ...]
    reason: str


class Router(Protocol):
    def route(self, probe: dict[str, Any]) -> RouteDecision:
        """Return a bounded routing decision."""


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
            dict.fromkeys(self._TARGETS[flag] for flag in flags if flag in self._TARGETS)
        )
        return RouteDecision(
            reinspect=bool(targets),
            reinspection_targets=targets,
            reason=";".join(flags) if flags else "low-risk agreement",
        )


class ApiRouter:
    """Use a short API probe, then validate it against the rule route."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.rule_router = RuleRouter()

    def route(self, probe: dict[str, Any]) -> RouteDecision:
        fallback = self.rule_router.route(probe)
        response = self.client.generate(
            ModelRequest(
                request_id="router-probe",
                model="fixture-model",
                system_prompt="Return only a concise routing JSON object.",
                user_prompt=str(probe),
            )
        )
        return RouteDecision(
            reinspect=fallback.reinspect,
            reinspection_targets=fallback.reinspection_targets,
            reason=f"api:{fallback.reason}:{response.provider}",
        )
