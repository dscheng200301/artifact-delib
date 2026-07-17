"""Bounded controlled cross-examination without hidden chain-of-thought."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from histodelib.api.base import ModelClient
from histodelib.api.response_parser import parse_json_object
from histodelib.methods.probe import RelationProbeResult
from histodelib.methods.reinspection import ReinspectionDecision
from histodelib.schemas import ModelRequest, TokenUsage


@dataclass(frozen=True)
class CrossExamResult:
    rounds: int
    stop_reason: str
    usage: TokenUsage = TokenUsage()
    api_calls: int = 0
    transcript: tuple[str, ...] = ()


class ControlledCrossExamination:
    """Stop on stability, abstention, or a configured maximum round count."""

    def __init__(self, max_rounds: int = 2) -> None:
        self.max_rounds = max(0, max_rounds)

    def run(self, probe: RelationProbeResult, decision: ReinspectionDecision) -> CrossExamResult:
        if not probe.risk_flags:
            return CrossExamResult(0, "stable")
        if not decision.targets:
            return CrossExamResult(0, "abstain")
        return CrossExamResult(self.max_rounds, "max_rounds")

    def run_with_client(
        self,
        client: ModelClient,
        probe: RelationProbeResult,
        decision: ReinspectionDecision,
        *,
        model_name: str,
    ) -> CrossExamResult:
        """Ask bounded, evidence-focused questions without exposing hidden reasoning."""

        if not probe.risk_flags:
            return CrossExamResult(0, "stable")
        if not decision.targets or self.max_rounds == 0:
            return CrossExamResult(0, "abstain")
        transcript: list[str] = []
        input_tokens = output_tokens = 0
        for round_no in range(self.max_rounds):
            response = client.generate(
                ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=model_name,
                    system_prompt=(
                        "You are a controlled cross-examiner. Ask one concise question and "
                        "return JSON with question, finding, and stop=true/false."
                    ),
                    user_prompt=(
                        f"Round {round_no + 1}; risk_flags={list(probe.risk_flags)}; "
                        f"targets={list(decision.targets)}"
                    ),
                    max_output_tokens=256,
                )
            )
            parsed = parse_json_object(response.content)
            transcript.append(str(parsed or {"raw": response.content}))
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            if parsed.get("stop") is True:
                return CrossExamResult(
                    round_no + 1,
                    "model_stop",
                    TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    round_no + 1,
                    tuple(transcript),
                )
        return CrossExamResult(
            self.max_rounds,
            "max_rounds",
            TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            self.max_rounds,
            tuple(transcript),
        )
