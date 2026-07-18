"""Explainable routing from short disagreement-probe features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from histodelib.api.base import ModelClient
from histodelib.constants import DEFAULT_MODEL, JSON_RESPONSE_SCHEMA
from histodelib.schemas import ModelRequest, TokenUsage


@dataclass(frozen=True)
class RouteDecision:
    """A bounded routing decision that never reads reference labels."""

    reinspect: bool
    reinspection_targets: tuple[str, ...]
    reason: str
    action: Literal["ACCEPT", "REINSPECT", "ABSTAIN"] = "ACCEPT"
    confidence: float | None = None
    reason_codes: tuple[str, ...] = ()
    source: Literal["rule", "api", "fallback"] = "rule"
    disagreement: bool = False

    @property
    def targets(self) -> tuple[str, ...]:
        return self.reinspection_targets


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
    api_calls = 0
    last_api_calls = 0
    last_usage = TokenUsage()

    def route(self, probe: dict[str, Any]) -> RouteDecision:
        flags = [str(flag) for flag in probe.get("risk_flags", [])]
        targets = tuple(
            dict.fromkeys(self._TARGETS[flag] for flag in flags if flag in self._TARGETS)
        )
        return RouteDecision(
            reinspect=bool(targets),
            reinspection_targets=targets,
            reason=";".join(flags) if flags else "low-risk agreement",
            action="REINSPECT" if targets else "ACCEPT",
            reason_codes=tuple(flags) if flags else ("stable",),
            source="rule",
        )


class ApiRouter:
    """Use a short API probe, then validate it against the rule route."""

    def __init__(self, client: ModelClient, model_name: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model_name = model_name
        self.rule_router = RuleRouter()
        self.api_calls = 0
        self.last_api_calls = 0
        self.last_usage = TokenUsage()

    def route(self, probe: dict[str, Any]) -> RouteDecision:
        fallback = self.rule_router.route(probe)
        response = self.client.generate(
            ModelRequest(
                request_id="router-probe",
                model=self.model_name,
                system_prompt="Return only a concise routing JSON object.",
                user_prompt=str(probe),
                response_schema=dict(JSON_RESPONSE_SCHEMA),
            )
        )
        self.api_calls += 1
        self.last_api_calls = 1
        self.last_usage = response.usage
        try:
            from histodelib.api.response_parser import parse_json_object

            parsed = parse_json_object(response.content)
            action = str(parsed.get("action"))
            targets_raw = parsed.get("targets", [])
            reason_codes_raw = parsed.get("reason_codes", [])
            confidence_raw = parsed.get("confidence")
            if action not in {"ACCEPT", "REINSPECT", "ABSTAIN"}:
                raise ValueError("invalid router action")
            if not isinstance(targets_raw, list) or any(
                str(target) not in {"patch", "glyph", "panor", "text"}
                for target in targets_raw
            ):
                raise ValueError("invalid router targets")
            if len(targets_raw) > 2:
                raise ValueError("router target limit exceeded")
            if not isinstance(reason_codes_raw, list):
                raise ValueError("invalid router reason codes")
            confidence = (
                float(confidence_raw)
                if isinstance(confidence_raw, (int, float, str))
                else None
            )
            if confidence is not None and not 0.0 <= confidence <= 1.0:
                raise ValueError("invalid router confidence")
            targets = tuple(dict.fromkeys(str(target) for target in targets_raw))
            api_reinspect = action == "REINSPECT" and bool(targets)
            disagreement = (
                api_reinspect != fallback.reinspect
                or targets != fallback.reinspection_targets
            )
            return RouteDecision(
                reinspect=api_reinspect,
                reinspection_targets=targets,
                reason=f"api:{response.provider}",
                action=action,  # type: ignore[arg-type]
                confidence=confidence,
                reason_codes=tuple(str(code) for code in reason_codes_raw),
                source="api",
                disagreement=disagreement,
            )
        except (TypeError, ValueError):
            return RouteDecision(
                reinspect=fallback.reinspect,
                reinspection_targets=fallback.reinspection_targets,
                reason=f"fallback:{fallback.reason}",
                action=fallback.action,
                confidence=fallback.confidence,
                reason_codes=tuple(dict.fromkeys((*fallback.reason_codes, "ROUTER_PARSE_FAILURE"))),
                source="fallback",
                disagreement=False,
            )
