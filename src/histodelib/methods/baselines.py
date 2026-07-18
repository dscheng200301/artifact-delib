"""API-only baseline schedules with explicit call accounting."""

from __future__ import annotations

import base64
import uuid
from collections import Counter
from typing import Protocol

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.constants import DEFAULT_MODEL, JSON_RESPONSE_SCHEMA, LABEL_JSON_INSTRUCTION
from histodelib.methods.histodelib import HistoDelibMethod
from histodelib.methods.router import ApiRouter, RuleRouter
from histodelib.schemas import Label, ModelRequest, Prediction, Sample, TokenUsage

BASELINE_NAMES = (
    "text_only",
    "image_only",
    "direct_vlm",
    "structured_reasoning",
    "self_consistency",
    "self_reflection",
    "sequential_context_veracity",
    "fixed_multi_perspective",
    "generic_mad",
    "always_full",
    "histodelib_rule",
    "histodelib_api_router",
)


class VerificationMethod(Protocol):
    def run(self, sample: Sample) -> Prediction:
        """Run one method on one sample."""


class SingleCallBaseline:
    def __init__(
        self,
        name: str,
        client: ModelClient,
        model_name: str = DEFAULT_MODEL,
        input_mode: str = "text",
    ) -> None:
        self.name = name
        self.client = client
        self.model_name = model_name
        self.input_mode = input_mode

    def run(self, sample: Sample) -> Prediction:
        response = self.client.generate(
            _build_request(self.name, self.model_name, sample, self.input_mode)
        )
        label = _read_label(response.content)
        return Prediction(
            sample_id=sample.sample_id,
            method=self.name,
            final_label=label,
            initial_label=label,
            status="COMPLETED" if label else "INSUFFICIENT_EVIDENCE",
            usage=response.usage,
            api_calls=1,
        )


class RepeatedBaseline:
    def __init__(
        self,
        name: str,
        client: ModelClient,
        calls: int,
        model_name: str = DEFAULT_MODEL,
        input_mode: str = "text",
    ) -> None:
        self.name = name
        self.client = client
        self.calls = calls
        self.model_name = model_name
        self.input_mode = input_mode

    def run(self, sample: Sample) -> Prediction:
        labels: list[Label] = []
        usage = TokenUsage()
        for _ in range(self.calls):
            response = self.client.generate(
                _build_request(
                    self.name,
                    self.model_name,
                    sample,
                    self.input_mode,
                    round_no=len(labels),
                )
            )
            label = _read_label(response.content)
            if label is not None:
                labels.append(label)
            usage = TokenUsage(
                input_tokens=usage.input_tokens + response.usage.input_tokens,
                output_tokens=usage.output_tokens + response.usage.output_tokens,
            )
        final = Counter(labels).most_common(1)[0][0] if labels else None
        return Prediction(
            sample_id=sample.sample_id,
            method=self.name,
            final_label=final,
            initial_label=labels[0] if labels else None,
            status="COMPLETED" if final else "INSUFFICIENT_EVIDENCE",
            usage=usage,
            api_calls=self.calls,
        )


def create_baseline(
    name: str,
    client: ModelClient,
    model_name: str = DEFAULT_MODEL,
    *,
    enable_api_deliberation: bool = False,
    max_reinspection_targets: int = 2,
    max_cross_exam_rounds: int = 2,
) -> VerificationMethod:
    """Create a named baseline while preserving every call in its prediction."""

    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {name}")
    if name == "histodelib_rule":
        return HistoDelibMethod(
            client=client,
            router=RuleRouter(),
            model_name=model_name,
            enable_api_deliberation=enable_api_deliberation,
            max_reinspection_targets=max_reinspection_targets,
            max_cross_exam_rounds=max_cross_exam_rounds,
        )
    if name == "histodelib_api_router":
        return HistoDelibMethod(
            client=client,
            router=ApiRouter(client, model_name=model_name),
            method_name="histodelib_api_router",
            model_name=model_name,
            enable_api_deliberation=enable_api_deliberation,
            max_reinspection_targets=max_reinspection_targets,
            max_cross_exam_rounds=max_cross_exam_rounds,
        )
    calls = {
        "self_consistency": 3,
        "self_reflection": 2,
        "sequential_context_veracity": 2,
        "fixed_multi_perspective": 3,
        "generic_mad": 4,
        "always_full": 6,
    }.get(name, 1)
    mode = "text"
    if name in {"image_only", "direct_vlm", "structured_reasoning", "always_full"}:
        mode = "image_and_text"
    if name == "image_only":
        mode = "image"
    if name == "always_full":
        mode = "full_schedule"
    return (
        RepeatedBaseline(name, client, calls, model_name, mode)
        if calls > 1
        else SingleCallBaseline(name, client, model_name, mode)
    )


def _build_request(
    name: str,
    model_name: str,
    sample: Sample,
    input_mode: str,
    *,
    round_no: int = 0,
) -> ModelRequest:
    if input_mode == "full_schedule":
        input_mode = ("text", "image", "image_and_text", "image", "text", "image_and_text")[
            min(round_no, 5)
        ]
    image_base64 = None
    user_prompt = sample.caption
    if input_mode in {"image", "image_and_text"}:
        encoded = base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
        image_base64 = f"data:image/png;base64,{encoded}"
    if input_mode == "image":
        user_prompt = "Analyze the supplied image only; return a JSON label."
    elif input_mode == "image_and_text":
        user_prompt = f"Caption: {sample.caption}\nAnalyze the supplied image and caption."
    protocol = {
        "text_only": "Use caption text only.",
        "image_only": "Use visible image evidence only.",
        "direct_vlm": "Assess caption and image in one direct pass.",
        "structured_reasoning": "Return label plus concise evidence fields.",
        "self_consistency": "Produce an independent sample for majority vote.",
        "self_reflection": "Review the previous pass and correct only clear errors.",
        "sequential_context_veracity": "Check caption context before image consistency.",
        "fixed_multi_perspective": "Use a fixed independent perspective for this pass.",
        "generic_mad": "Act as one independent debate member.",
        "always_full": "Execute the configured full protocol stage.",
    }.get(name, "Return a JSON label.")
    return ModelRequest(
        request_id=str(uuid.uuid4()),
        model=model_name,
        system_prompt=f"Baseline {name}, round {round_no}. {protocol} {LABEL_JSON_INSTRUCTION}",
        user_prompt=user_prompt,
        image_base64=image_base64,
        response_schema=dict(JSON_RESPONSE_SCHEMA),
    )


def _read_label(content: str) -> Label | None:
    try:
        return Label(str(parse_json_object(content).get("label")))
    except (TypeError, ValueError):
        return None
