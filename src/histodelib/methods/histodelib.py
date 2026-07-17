"""A bounded disagreement-triggered orchestration path for fixture validation."""

from __future__ import annotations

import uuid

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.methods.judge import DeferredJudge
from histodelib.methods.router import RuleRouter
from histodelib.schemas import Label, ModelRequest, Prediction, Sample, TokenUsage


class HistoDelibMethod:
    """Isolate caption and image views, route disagreement, then defer judgment."""

    def __init__(
        self,
        client: ModelClient,
        router: RuleRouter,
        judge: DeferredJudge | None = None,
    ) -> None:
        self.client = client
        self.router = router
        self.judge = judge or DeferredJudge()

    def run(self, sample: Sample) -> Prediction:
        text_response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt="You are the Text Agent. Analyze only the caption claim.",
                user_prompt=sample.caption,
            )
        )
        image_response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model="fixture-model",
                system_prompt="You are the Image Agent. Analyze only visible image evidence.",
                user_prompt=f"Image path: {sample.image_path.name}",
            )
        )
        text_label = self._read_label(text_response.content)
        image_label = self._read_label(image_response.content)
        risk_flags = [] if text_label == image_label else ["modality_disagreement"]
        route = self.router.route({"risk_flags": risk_flags})
        judged = self.judge.adjudicate(
            image_label,
            {"text_label": text_label, "image_label": image_label},
        )
        usage = TokenUsage(
            input_tokens=text_response.usage.input_tokens + image_response.usage.input_tokens,
            output_tokens=text_response.usage.output_tokens + image_response.usage.output_tokens,
        )
        return Prediction(
            sample_id=sample.sample_id,
            method="histodelib_rule",
            initial_label=image_label,
            final_label=judged.final_label,
            status="COMPLETED" if judged.final_label is not None else "INSUFFICIENT_EVIDENCE",
            evidence={
                "text_label": text_label.value if text_label else None,
                "image_label": image_label.value if image_label else None,
                "route": route.reason,
                "reinspection_targets": route.reinspection_targets,
                "judge_decision": judged.decision,
            },
            usage=usage,
            api_calls=2,
        )

    @staticmethod
    def _read_label(content: str) -> Label | None:
        try:
            return Label(str(parse_json_object(content).get("label")))
        except (ValueError, TypeError):
            return None
