"""Tests for ArtifactDelib Phase 2 — routing and targeted recheck."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.router.rule_router import RuleRouter
from artifact_delib.schemas import (
    ArtifactCandidate,
    CandidateSet,
    DisagreementAnalysis,
    RouteDecision,
)
from artifact_delib.utils.fixture_builder import build_artifact_fixtures
from artifact_delib.api.schemas import ModelResponse, TokenUsage


# ═══════════════════════════════════════════════
#  RuleRouter unit tests
# ═══════════════════════════════════════════════

@pytest.fixture
def router():
    return RuleRouter(max_recheck_rounds=2)


@pytest.fixture
def high_conf_candidates():
    return CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.85),
        ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.10),
    ))


@pytest.fixture
def low_conf_candidates():
    return CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.48),
        ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.32),
        ArtifactCandidate(text="明代早期青花梅瓶", confidence=0.20),
    ))


@pytest.fixture
def style_disagreement():
    return DisagreementAnalysis(
        content="分歧集中在年代判断。",
        route_hint="STYLE",
    )


def test_router_fast_on_high_confidence(
    router: RuleRouter,
    high_conf_candidates: CandidateSet,
) -> None:
    decision = router.route(
        disagreement=None,
        candidates=high_conf_candidates,
    )
    assert decision.action == "FAST"
    assert "high confidence" in decision.reason


def test_router_style_recheck_on_style_disagreement(
    router: RuleRouter,
    low_conf_candidates: CandidateSet,
    style_disagreement: DisagreementAnalysis,
) -> None:
    decision = router.route(
        disagreement=style_disagreement,
        candidates=low_conf_candidates,
    )
    assert decision.action == "STYLE_RECHECK"
    assert decision.recheck_count == 0


def test_router_does_not_repeat_recheck_type(
    router: RuleRouter,
    low_conf_candidates: CandidateSet,
    style_disagreement: DisagreementAnalysis,
) -> None:
    """After STYLE_RECHECK is done, should pick a different type."""
    decision = router.route(
        disagreement=style_disagreement,
        candidates=low_conf_candidates,
        recheck_count=1,
        completed_rechecks=("STYLE_RECHECK",),
    )
    assert decision.action != "STYLE_RECHECK"
    assert "already done" in decision.reason


def test_router_deliberation_after_max_recheck(
    router: RuleRouter,
    low_conf_candidates: CandidateSet,
    style_disagreement: DisagreementAnalysis,
) -> None:
    """After max recheck rounds with low margin → DELIBERATION."""
    decision = router.route(
        disagreement=style_disagreement,
        candidates=low_conf_candidates,
        recheck_count=2,
        completed_rechecks=("STYLE_RECHECK", "LOCAL_DETAIL_RECHECK"),
    )
    assert decision.action == "DELIBERATION"
    assert "uncertain" in decision.reason


def test_router_fast_after_deliberation(
    router: RuleRouter,
    low_conf_candidates: CandidateSet,
) -> None:
    """If deliberation already done, always FAST."""
    decision = router.route(
        disagreement=None,
        candidates=low_conf_candidates,
        deliberation_count=1,
    )
    assert decision.action == "FAST"


# ═══════════════════════════════════════════════
#  Pipeline integration tests
# ═══════════════════════════════════════════════

@pytest.fixture
def sample():
    with tempfile.TemporaryDirectory() as tmp:
        samples = build_artifact_fixtures(Path(tmp))
        yield samples[0]


def test_pipeline_runs_full_routing(sample) -> None:
    """Full pipeline with routing should complete and produce route decisions."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=2,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    assert result.status == "COMPLETED"
    assert len(result.route_decisions) >= 1
    assert result.final_identification.content
    assert len(result.expert_reports) == 5


def test_pipeline_initial_candidates_preserved(sample) -> None:
    """Initial candidates should reflect the pre-routing state."""
    pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
    result = pipeline.run(sample.image_path, sample.sample_id)

    assert len(result.initial_candidates.candidates) == 3
    assert result.initial_candidates.candidates[0].confidence == 0.48


def test_pipeline_recheck_not_duplicated(sample) -> None:
    """Recheck types should not repeat."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=3,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    recheck_actions = [
        rd.action for rd in result.route_decisions if "RECHECK" in rd.action
    ]
    assert len(set(recheck_actions)) == len(recheck_actions), (
        f"Recheck types repeated: {recheck_actions}"
    )


def test_pipeline_with_high_confidence_skips_recheck(sample) -> None:
    """High-confidence candidates should go FAST with no recheck."""

    class HighConfMock:
        def generate(self, request):
            pn = (request.prompt_name or "").lower()
            if "candidate" in pn:
                content = (
                    '```json\n{"candidates": ['
                    '{"text": "明永乐青花梅瓶", "confidence": 0.85},'
                    '{"text": "明宣德青花梅瓶", "confidence": 0.10}]}\n```'
                )
            elif "disagreement" in pn:
                content = "无明显分歧。\n分歧类型：MULTI_FACTOR"
            else:
                content = "观察描述。"
            return ModelResponse(
                request_id="x", content=content,
                usage=TokenUsage(input_tokens=5, output_tokens=5),
                latency_ms=0, provider="mock", model="m",
            )

    pipeline = ArtifactDelibPipeline(client=HighConfMock())
    result = pipeline.run(sample.image_path, sample.sample_id)

    assert result.route_decisions[0].action == "FAST"
    assert len(result.recheck_reports) == 0


def test_pipeline_updated_candidates_after_recheck(sample) -> None:
    """After recheck, candidates should have updated confidence."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=2,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    if result.updated_candidates:
        # Updated confidence should differ from initial
        initial_conf = result.initial_candidates.candidates[0].confidence
        updated_conf = result.updated_candidates.candidates[0].confidence
        assert updated_conf != initial_conf


def test_pipeline_disagreement_analysis_present(sample) -> None:
    """Disagreement analysis should be present in the result."""
    pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
    result = pipeline.run(sample.image_path, sample.sample_id)

    assert result.disagreement_analysis is not None
    assert result.disagreement_analysis.route_hint in {
        "SHAPE", "STYLE", "GLYPH", "MATERIAL", "LOCAL_DETAIL", "MULTI_FACTOR"
    }


def test_pipeline_tracks_api_calls(sample) -> None:
    """API call count should be correctly accumulated."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=2,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    # Each recheck loop adds: 1 expert + 1 summarizer + 1 candidate + 1 disagreement = 4
    # Initial: 1 VP + 5 experts + 1 summarizer + 1 candidate + 1 disagreement = 9
    # Final: 1 judge
    # Total with 2 rechecks: 9 + 4*2 + 1 = 18
    assert result.total_api_calls >= 10  # at least initial + judge
    assert result.total_api_calls <= 9 + 4 * 3 + 1  # max with 3 recheck attempts
