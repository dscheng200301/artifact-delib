"""Deferred adjudication over concise structured evidence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.constants import JSON_RESPONSE_SCHEMA
from histodelib.schemas import Label, ModelRequest, TokenUsage


@dataclass(frozen=True)
class JudgeResult:
    decision: Literal["KEEP", "REVISE", "ABSTAIN"]
    final_label: Label | None
    usage: TokenUsage = TokenUsage()
    api_calls: int = 0
    fallback_reason: str | None = None
    schema_version: str = "judge-v1"


class DeferredJudge:
    """Keep a blind label unless concise evidence warrants a revision."""

    def adjudicate(self, blind_label: Label | None, evidence: dict[str, Any]) -> JudgeResult:
        text_label = self._label(evidence.get("text_label"))
        image_label = self._label(evidence.get("image_label"))
        if text_label is None and image_label is None:
            return JudgeResult("ABSTAIN", blind_label)
        if text_label is not None and text_label == image_label:
            if text_label == blind_label:
                return JudgeResult("KEEP", blind_label)
            return JudgeResult("REVISE", text_label)
        if text_label is not None and text_label is not Label.TRUE:
            return JudgeResult("REVISE", text_label)
        return JudgeResult("KEEP", blind_label)

    def adjudicate_with_client(
        self,
        client: ModelClient,
        blind_label: Label | None,
        evidence: dict[str, Any],
        *,
        model_name: str,
    ) -> JudgeResult:
        """Use one bounded API call to adjudicate concise, non-chain-of-thought evidence."""

        response = client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model=model_name,
                system_prompt=(
                    "You are a deferred judge. Return JSON only with decision KEEP, REVISE, "
                    "or ABSTAIN and an optional final_label. Output only valid JSON."
                ),
                user_prompt=f"blind_label={blind_label}; evidence={evidence}",
                max_output_tokens=256,
                response_schema=dict(JSON_RESPONSE_SCHEMA),
            )
        )
        parsed = parse_json_object(response.content)
        candidate = self._label(parsed.get("label") or parsed.get("final_label"))
        evidence_labels = [
            self._label(evidence.get("text_label")),
            self._label(evidence.get("image_label")),
        ]
        if (
            candidate is Label.TRUE
            and evidence_labels[0] is not None
            and evidence_labels[0] == evidence_labels[1]
            and evidence_labels[0] is not Label.TRUE
        ):
            candidate = evidence_labels[0]
        if candidate is None:
            fallback = self.adjudicate(blind_label, evidence)
            return JudgeResult(
                fallback.decision,
                fallback.final_label,
                response.usage,
                1,
                fallback_reason="JUDGE_PARSE_FAILURE_FALLBACK",
            )
        decision: Literal["KEEP", "REVISE"] = "KEEP" if candidate == blind_label else "REVISE"
        return JudgeResult(decision, candidate, response.usage, 1)

    @staticmethod
    def _label(value: object) -> Label | None:
        try:
            return Label(str(value))
        except ValueError:
            return None
