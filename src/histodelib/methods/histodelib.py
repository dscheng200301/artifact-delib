"""A bounded disagreement-triggered orchestration path for fixture validation."""

from __future__ import annotations

from histodelib.api.base import ModelClient
from histodelib.methods.agents import ImageAgent, TextAgent
from histodelib.methods.cross_exam import ControlledCrossExamination
from histodelib.methods.judge import DeferredJudge
from histodelib.methods.probe import LightRelationProbe
from histodelib.methods.reinspection import TargetedReinspection
from histodelib.methods.router import Router
from histodelib.schemas import Prediction, Sample, TokenUsage


class HistoDelibMethod:
    """Isolate caption and image views, route disagreement, then defer judgment."""

    def __init__(
        self,
        client: ModelClient,
        router: Router,
        judge: DeferredJudge | None = None,
        method_name: str = "histodelib_rule",
    ) -> None:
        self.client = client
        self.router = router
        self.judge = judge or DeferredJudge()
        self.method_name = method_name

    def run(self, sample: Sample) -> Prediction:
        text_evidence = TextAgent(self.client).analyze(sample.caption)
        image_evidence = ImageAgent(self.client).analyze(sample.image_path)
        text_label = text_evidence.label
        image_label = image_evidence.label
        probe = LightRelationProbe().assess(
            {"label": text_label.value if text_label else None, "claims": [sample.caption]},
            {"label": image_label.value if image_label else None},
        )
        reinspection = TargetedReinspection().select(probe)
        cross_exam = ControlledCrossExamination().run(probe, reinspection)
        route = self.router.route({"risk_flags": list(probe.risk_flags)})
        judged = self.judge.adjudicate(
            image_label,
            {"text_label": text_label, "image_label": image_label},
        )
        usage = TokenUsage(
            input_tokens=(
                text_evidence.usage.input_tokens
                + image_evidence.usage.input_tokens
                + getattr(self.router, "last_usage", TokenUsage()).input_tokens
            ),
            output_tokens=(
                text_evidence.usage.output_tokens
                + image_evidence.usage.output_tokens
                + getattr(self.router, "last_usage", TokenUsage()).output_tokens
            ),
        )
        return Prediction(
            sample_id=sample.sample_id,
            method=self.method_name,
            initial_label=image_label,
            final_label=judged.final_label,
            status="COMPLETED" if judged.final_label is not None else "INSUFFICIENT_EVIDENCE",
            evidence={
                "text_label": text_label.value if text_label else None,
                "image_label": image_label.value if image_label else None,
                "route": route.reason,
                "reinspection_targets": route.reinspection_targets,
                "probe_flags": probe.risk_flags,
                "cross_exam_rounds": cross_exam.rounds,
                "cross_exam_stop_reason": cross_exam.stop_reason,
                "judge_decision": judged.decision,
            },
            usage=usage,
            api_calls=2 + getattr(
                self.router,
                "last_api_calls",
                getattr(self.router, "api_calls", 0),
            ),
        )
