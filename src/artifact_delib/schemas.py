"""Typed data structures for the ArtifactDelib framework.

Natural language is the primary carrier — these schemas provide lightweight
structure for the pipeline while keeping NL as the main content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from artifact_delib.api.schemas import TokenUsage  # noqa: F401 — reuse token tracking


@dataclass(frozen=True)
class ArtifactSample:
    """A single artifact identification input — image only during inference."""

    sample_id: str
    image_path: Path
    # Gold label — only for training/evaluation, NEVER passed to inference modules
    category: str | None = None
    fine_grained_type: str | None = None
    period: str | None = None
    material: str | None = None
    craft: str | None = None
    dynasty: str | None = None
    region: str | None = None
    source: str | None = None
    split: Literal["train", "validation", "test", "fixture", "unassigned"] | None = None
    artifact_group_id: str | None = None


@dataclass(frozen=True)
class ExpertReport:
    """One expert's natural-language analysis of an artifact image."""

    expert_name: str
    content: str  # Natural language expert opinion (100-200 chars)
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class VisualPerceptionReport:
    """Initial visual perception — overall description only."""

    content: str  # 100-200 chars NL
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class SummarizedReport:
    """Compressed multi-expert summary — no structured fields."""

    content: str  # 200-400 chars NL
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ArtifactCandidate:
    """One candidate artifact identity — uses NL text as primary identifier."""

    text: str       # e.g. "明永乐青花梅瓶"
    confidence: float  # 0.0–1.0


@dataclass(frozen=True)
class CandidateSet:
    """Top-K artifact candidates in ranked order."""

    candidates: tuple[ArtifactCandidate, ...]
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def top1(self) -> ArtifactCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top2(self) -> ArtifactCandidate | None:
        return self.candidates[1] if len(self.candidates) > 1 else None

    @property
    def top1_confidence(self) -> float:
        return self.top1.confidence if self.top1 else 0.0

    @property
    def top2_confidence(self) -> float:
        return self.top2.confidence if self.top2 else 0.0

    @property
    def margin(self) -> float:
        return self.top1_confidence - self.top2_confidence


@dataclass(frozen=True)
class DisagreementAnalysis:
    """Natural-language analysis of why candidates differ, plus a route hint."""

    content: str  # NL analysis
    route_hint: Literal[
        "SHAPE", "STYLE", "GLYPH", "MATERIAL", "LOCAL_DETAIL", "MULTI_FACTOR"
    ] = "MULTI_FACTOR"
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class DeliberationRound:
    """One round of controlled deliberation between candidate hypotheses."""

    round_no: int
    candidate_a_opinion: str
    candidate_b_opinion: str
    candidate_a_decision: Literal["MAINTAIN", "REVISE", "ABSTAIN"]
    candidate_b_decision: Literal["MAINTAIN", "REVISE", "ABSTAIN"]
    critic_feedback: str


@dataclass(frozen=True)
class DeliberationResult:
    """Result of controlled deliberation between hypotheses."""

    rounds: tuple[DeliberationRound, ...]
    stop_reason: str
    summary: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    total_api_calls: int = 0
@dataclass(frozen=True)
class FinalIdentification:
    """Final natural-language identification from the deferred judge."""

    content: str  # NL paragraph
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class RecheckRecord:
    """A single targeted recheck — which expert, what context, what changed."""

    round_no: int
    expert_name: str
    previous_content: str  # The expert's report before recheck
    new_content: str       # The expert's report after recheck
    context_query: str     # The disagreement context that triggered this recheck
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class RouteDecision:
    """Router decision — which path to take."""

    action: Literal[
        "FAST",
        "SHAPE_RECHECK",
        "STYLE_RECHECK",
        "GLYPH_RECHECK",
        "MATERIAL_RECHECK",
        "LOCAL_DETAIL_RECHECK",
        "DELIBERATION",
    ]
    reason: str
    recheck_count: int = 0
    deliberation_count: int = 0


@dataclass(frozen=True)
class PipelineResult:
    """Complete pipeline result with full provenance."""

    sample_id: str
    final_identification: FinalIdentification
    visual_perception_report: VisualPerceptionReport
    expert_reports: tuple[ExpertReport, ...]
    summarized_report: SummarizedReport
    initial_candidates: CandidateSet
    disagreement_analysis: DisagreementAnalysis | None = None
    route_decisions: tuple[RouteDecision, ...] = ()
    recheck_reports: tuple[ExpertReport, ...] = ()
    updated_candidates: CandidateSet | None = None
    recheck_records: tuple[RecheckRecord, ...] = ()
    deliberation_result: DeliberationResult | None = None
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    total_api_calls: int = 0
    status: str = "COMPLETED"
