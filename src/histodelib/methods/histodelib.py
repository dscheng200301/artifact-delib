"""A bounded disagreement-triggered orchestration path for fixture validation."""

from __future__ import annotations

from typing import Any

from histodelib.api.base import ModelClient
from histodelib.constants import DEFAULT_MODEL
from histodelib.methods.agents import ImageAgent, TextAgent
from histodelib.methods.cross_exam import ControlledCrossExamination
from histodelib.methods.judge import DeferredJudge
from histodelib.methods.probe import LightRelationProbe
from histodelib.methods.reinspection import TargetedReinspection
from histodelib.methods.router import Router
from histodelib.schemas import ImageEvidence, Prediction, Sample, TextEvidence, TokenUsage


class HistoDelibMethod:
    """Isolate caption and image views, route disagreement, then defer judgment."""

    def __init__(
        self,
        client: ModelClient,
        router: Router,
        judge: DeferredJudge | None = None,
        method_name: str = "histodelib_rule",
        model_name: str = DEFAULT_MODEL,
        enable_api_deliberation: bool = False,
        max_reinspection_targets: int = 2,
        max_cross_exam_rounds: int = 2,
    ) -> None:
        self.client = client
        self.router = router
        self.judge = judge or DeferredJudge()
        self.method_name = method_name
        self.model_name = model_name
        self.enable_api_deliberation = enable_api_deliberation
        self.max_reinspection_targets = max(0, max_reinspection_targets)
        self.max_cross_exam_rounds = max(0, max_cross_exam_rounds)

    def run(self, sample: Sample) -> Prediction:
        text_evidence = TextAgent(self.client, model_name=self.model_name).analyze(sample.caption)
        image_evidence = ImageAgent(self.client, model_name=self.model_name).analyze(
            sample.image_path
        )
        text_label = text_evidence.label
        image_label = image_evidence.label
        text_structured = text_evidence.structured
        image_structured = image_evidence.structured
        if not isinstance(text_structured, TextEvidence) or not isinstance(
            image_structured, ImageEvidence
        ):
            raise TypeError("modal agents returned mismatched evidence types")
        text_probe = {
            "label": text_label.value if text_label else None,
            "claims": list(text_structured.caption_claims),
            "requires_visible_text": text_structured.requires_visible_text,
        }
        image_probe = {
            "label": image_label.value if image_label else None,
            "visible_text": image_structured.visible_text,
            "region_candidates": image_structured.region_candidates,
        }
        probe = LightRelationProbe().assess(
            text_probe,
            image_probe,
        )
        reinspection = TargetedReinspection().select(probe)
        route = self.router.route({"risk_flags": list(probe.risk_flags)})
        evidence_for_judge: dict[str, Any] = {
            "text_label": text_label,
            "image_label": image_label,
            "text_evidence": text_structured.model_dump(mode="json"),
            "image_evidence": image_structured.model_dump(mode="json"),
        }
        if self.enable_api_deliberation and route.reinspect:
            reinspection_result = TargetedReinspection().inspect(
                self.client,
                sample,
                reinspection,
                model_name=self.model_name,
                max_targets=self.max_reinspection_targets,
            )
            cross_exam = ControlledCrossExamination(
                max_rounds=self.max_cross_exam_rounds
            ).run_with_client(
                self.client,
                probe,
                reinspection,
                model_name=self.model_name,
            )
            evidence_for_judge["reinspection"] = reinspection_result.evidence
            evidence_for_judge["cross_exam"] = cross_exam.transcript
            judged = self.judge.adjudicate_with_client(
                self.client,
                image_label,
                evidence_for_judge,
                model_name=self.model_name,
            )
        else:
            reinspection_result = None
            cross_exam = ControlledCrossExamination().run(probe, reinspection)
            judged = self.judge.adjudicate(image_label, evidence_for_judge)
        usage = TokenUsage(
            input_tokens=(
                text_evidence.usage.input_tokens
                + image_evidence.usage.input_tokens
                + getattr(self.router, "last_usage", TokenUsage()).input_tokens
                + (reinspection_result.usage.input_tokens if reinspection_result else 0)
                + cross_exam.usage.input_tokens
                + judged.usage.input_tokens
            ),
            output_tokens=(
                text_evidence.usage.output_tokens
                + image_evidence.usage.output_tokens
                + getattr(self.router, "last_usage", TokenUsage()).output_tokens
                + (reinspection_result.usage.output_tokens if reinspection_result else 0)
                + cross_exam.usage.output_tokens
                + judged.usage.output_tokens
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
                "route_action": route.action,
                "route_source": route.source,
                "route_reason_codes": route.reason_codes,
                "route_disagreement": route.disagreement,
                "reinspection_targets": route.reinspection_targets,
                "probe_flags": probe.risk_flags,
                "cross_exam_rounds": cross_exam.rounds,
                "cross_exam_stop_reason": cross_exam.stop_reason,
                "cross_exam_state": (
                    {
                        "initial_disagreement": cross_exam.state.initial_disagreement,
                        "rounds": [
                            {
                                "round_no": turn.round_no,
                                "question": turn.question,
                                "respondent": turn.respondent,
                                "answer": turn.answer,
                                "cited_evidence_ids": turn.cited_evidence_ids,
                                "resolved": turn.resolved,
                            }
                            for turn in (cross_exam.state.rounds if cross_exam.state else ())
                        ],
                        "stop_reason": cross_exam.state.stop_reason,
                    }
                    if cross_exam.state
                    else None
                ),
                "judge_decision": judged.decision,
                "judge_fallback_reason": judged.fallback_reason,
                "judge_schema_version": judged.schema_version,
                "reinspection_api_calls": (
                    reinspection_result.api_calls if reinspection_result else 0
                ),
                "cross_exam_api_calls": cross_exam.api_calls,
                "judge_api_calls": judged.api_calls,
            },
            usage=usage,
            api_calls=2 + getattr(
                self.router,
                "last_api_calls",
                getattr(self.router, "api_calls", 0),
            )
            + (reinspection_result.api_calls if reinspection_result else 0)
            + cross_exam.api_calls
            + judged.api_calls,
        )
