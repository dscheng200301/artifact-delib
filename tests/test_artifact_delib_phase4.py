"""Tests for ArtifactDelib Phase 4 — Controlled Expert Deliberation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.agents.deliberation.critic_agent import CriticAgent
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent
from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import (
    ArtifactCandidate,
    CandidateSet,
    DeliberationResult,
    ExpertReport,
    SummarizedReport,
)
from artifact_delib.utils.fixture_builder import build_artifact_fixtures
from artifact_delib.api.schemas import TokenUsage


# ═══════════════════════════════════════════════
#  HypothesisAgent tests
# ═══════════════════════════════════════════════

@pytest.fixture
def sample():
    with tempfile.TemporaryDirectory() as tmp:
        yield build_artifact_fixtures(Path(tmp))[0]


@pytest.fixture
def hypothesis_agent():
    return HypothesisAgent(client=ArtifactMockClient())


@pytest.fixture
def critic_agent():
    return CriticAgent(client=ArtifactMockClient())


@pytest.fixture
def mock_summary():
    return SummarizedReport(
        content="综合观察，该器物为一件青花瓷瓶，整体风格接近明代早期。",
    )


@pytest.fixture
def mock_expert_reports():
    return (
        ExpertReport(expert_name="shape", content="梅瓶器形，小口短颈丰肩。", usage=TokenUsage()),
        ExpertReport(expert_name="style", content="缠枝莲纹，明代早期风格。", usage=TokenUsage()),
        ExpertReport(expert_name="glyph", content="底部有款识区域。", usage=TokenUsage()),
        ExpertReport(expert_name="material", content="瓷质，青花釉下彩。", usage=TokenUsage()),
        ExpertReport(expert_name="local_detail", content="圈足有火石红。", usage=TokenUsage()),
    )


def test_hypothesis_agent_returns_opinion_and_decision(
    hypothesis_agent: HypothesisAgent,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """HypothesisAgent should return (opinion, decision) tuple."""
    opinion, decision = hypothesis_agent.argue(
        candidate_text="明永乐青花梅瓶",
        candidate_confidence=0.48,
        opponent_text="明宣德青花梅瓶",
        opponent_confidence=0.32,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
        round_no=1,
    )
    assert opinion
    assert decision in ("MAINTAIN", "REVISE", "ABSTAIN")


def test_hypothesis_agent_opinion_not_empty(
    hypothesis_agent: HypothesisAgent,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """Opinion text should not be empty."""
    opinion, _ = hypothesis_agent.argue(
        candidate_text="明永乐青花梅瓶",
        candidate_confidence=0.48,
        opponent_text="明宣德青花梅瓶",
        opponent_confidence=0.32,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
        round_no=1,
    )
    assert len(opinion) > 20


# ═══════════════════════════════════════════════
#  CriticAgent tests
# ═══════════════════════════════════════════════

def test_critic_agent_returns_feedback_and_continue_flag(
    critic_agent: CriticAgent,
) -> None:
    """Critic should return (feedback, should_continue)."""
    feedback, should_continue = critic_agent.evaluate(
        candidate_a_text="明永乐青花梅瓶",
        candidate_b_text="明宣德青花梅瓶",
        round_no=1,
        opinion_a="支持永乐时期的判断。",
        opinion_b="纹饰特征也存在支持宣德的因素。",
        decision_a="MAINTAIN",
        decision_b="MAINTAIN",
    )
    assert feedback
    assert isinstance(should_continue, bool)


# ═══════════════════════════════════════════════
#  DeliberationManager tests
# ═══════════════════════════════════════════════

@pytest.fixture
def manager():
    client = ArtifactMockClient()
    return DeliberationManager(
        hypothesis_agent=HypothesisAgent(client),
        critic_agent=CriticAgent(client),
        max_rounds=2,
    )


@pytest.fixture
def candidates():
    return CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.48),
        ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.32),
    ))


def test_deliberation_manager_returns_result(
    manager: DeliberationManager,
    candidates: CandidateSet,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """DeliberationManager.deliberate() should return DeliberationResult."""
    result = manager.deliberate(
        candidates=candidates,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
    )
    assert isinstance(result, DeliberationResult)
    assert result.stop_reason
    assert result.summary


def test_deliberation_rounds_produced(
    manager: DeliberationManager,
    candidates: CandidateSet,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """Deliberation should produce at least 1 round."""
    result = manager.deliberate(
        candidates=candidates,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
    )
    assert len(result.rounds) >= 1


def test_deliberation_round_has_all_fields(
    manager: DeliberationManager,
    candidates: CandidateSet,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """Each deliberation round should have all required fields."""
    result = manager.deliberate(
        candidates=candidates,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
    )
    for dr in result.rounds:
        assert dr.round_no >= 1
        assert dr.candidate_a_opinion
        assert dr.candidate_a_decision in ("MAINTAIN", "REVISE", "ABSTAIN")
        assert dr.candidate_b_decision in ("MAINTAIN", "REVISE", "ABSTAIN")
        assert dr.critic_feedback


def test_deliberation_respects_max_rounds(
    manager: DeliberationManager,
    candidates: CandidateSet,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """Deliberation should not exceed max_rounds."""
    result = manager.deliberate(
        candidates=candidates,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
    )
    assert len(result.rounds) <= 2


def test_deliberation_with_single_candidate(
    manager: DeliberationManager,
    mock_summary: SummarizedReport,
    mock_expert_reports: tuple[ExpertReport, ...],
) -> None:
    """With only one candidate, deliberation should handle gracefully."""
    single = CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.85),
    ))
    result = manager.deliberate(
        candidates=single,
        summarized_report=mock_summary,
        expert_reports=mock_expert_reports,
        recheck_reports=(),
    )
    assert result.stop_reason == "insufficient_candidates"
    assert len(result.rounds) == 0


# ═══════════════════════════════════════════════
#  Integration: Pipeline deliberation
# ═══════════════════════════════════════════════

def test_pipeline_deliberation_triggered_when_uncertain(sample) -> None:
    """Pipeline should trigger deliberation when candidates are uncertain
    after max recheck rounds."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=3,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    # Check if any route decision was DELIBERATION
    has_deliberation = any(
        rd.action == "DELIBERATION" for rd in result.route_decisions
    )
    # All recheck types may be exhausted first, so deliberation may or may not trigger
    # The key is: if deliberation was triggered, the result should be populated
    if has_deliberation:
        assert result.deliberation_result is not None
        assert len(result.deliberation_result.rounds) >= 1
    else:
        # If no deliberation, the pipeline should have reached FAST
        assert result.route_decisions[-1].action == "FAST"


def test_pipeline_deliberation_result_in_final_judge(sample) -> None:
    """When deliberation occurs, the judge should receive the deliberation summary."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=3,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    if result.deliberation_result is not None:
        # Final identification should be present
        assert result.final_identification
        assert len(result.final_identification.content) > 20
