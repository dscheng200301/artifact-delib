"""API-only baseline schedules with explicit call accounting."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Protocol

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.constants import DEFAULT_MODEL
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
    def __init__(self, name: str, client: ModelClient) -> None:
        self.name = name
        self.client = client

    def run(self, sample: Sample) -> Prediction:
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid.uuid4()),
                model=DEFAULT_MODEL,
                system_prompt=f"Baseline {self.name}. Return a JSON label.",
                user_prompt=sample.caption,
            )
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
    def __init__(self, name: str, client: ModelClient, calls: int) -> None:
        self.name = name
        self.client = client
        self.calls = calls

    def run(self, sample: Sample) -> Prediction:
        labels: list[Label] = []
        usage = TokenUsage()
        for _ in range(self.calls):
            response = self.client.generate(
                ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=DEFAULT_MODEL,
                    system_prompt=f"Baseline {self.name}. Return a JSON label.",
                    user_prompt=sample.caption,
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


def create_baseline(name: str, client: ModelClient) -> VerificationMethod:
    """Create a named baseline while preserving every call in its prediction."""

    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {name}")
    if name == "histodelib_rule":
        return HistoDelibMethod(client=client, router=RuleRouter())
    if name == "histodelib_api_router":
        return HistoDelibMethod(
            client=client, router=ApiRouter(client), method_name="histodelib_api_router"
        )
    calls = {
        "self_consistency": 3,
        "self_reflection": 2,
        "sequential_context_veracity": 2,
        "fixed_multi_perspective": 3,
        "generic_mad": 4,
        "always_full": 6,
    }.get(name, 1)
    return RepeatedBaseline(name, client, calls) if calls > 1 else SingleCallBaseline(name, client)


def _read_label(content: str) -> Label | None:
    try:
        return Label(str(parse_json_object(content).get("label")))
    except (TypeError, ValueError):
        return None
