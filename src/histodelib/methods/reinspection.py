"""Targeted reinspection selection."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from typing import Any

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.constants import JSON_RESPONSE_SCHEMA
from histodelib.methods.probe import RelationProbeResult
from histodelib.schemas import ModelRequest, Sample, TokenUsage


@dataclass(frozen=True)
class ReinspectionDecision:
    targets: tuple[str, ...]


@dataclass(frozen=True)
class ReinspectionResult:
    targets: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...] = ()
    usage: TokenUsage = TokenUsage()
    api_calls: int = 0


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

    def inspect(
        self,
        client: ModelClient,
        sample: Sample,
        decision: ReinspectionDecision,
        *,
        model_name: str,
        max_targets: int = 2,
    ) -> ReinspectionResult:
        """Inspect only routed image views, with an explicit call cap."""

        targets = decision.targets[: max(0, max_targets)]
        if not targets:
            return ReinspectionResult(())
        encoded = base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
        records: list[dict[str, Any]] = []
        input_tokens = output_tokens = 0
        for target in targets:
            response = client.generate(
                ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=model_name,
                    system_prompt=(
                        "You are a targeted visual reinspection agent. Return concise JSON "
                        "with label and evidence; inspect only the requested view. "
                        "Output only valid JSON."
                    ),
                    user_prompt=f"Reinspect view '{target}' for sample {sample.sample_id}.",
                    image_base64=f"data:image/png;base64,{encoded}",
                    max_output_tokens=256,
                    response_schema=dict(JSON_RESPONSE_SCHEMA),
                )
            )
            parsed = parse_json_object(response.content)
            records.append({"target": target, "response": parsed or {"raw": response.content}})
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
        return ReinspectionResult(
            targets=targets,
            evidence=tuple(records),
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            api_calls=len(targets),
        )
