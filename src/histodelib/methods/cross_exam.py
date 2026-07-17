"""Bounded controlled cross-examination without hidden chain-of-thought."""

from __future__ import annotations

from dataclasses import dataclass

from histodelib.methods.probe import RelationProbeResult
from histodelib.methods.reinspection import ReinspectionDecision


@dataclass(frozen=True)
class CrossExamResult:
    rounds: int
    stop_reason: str


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
