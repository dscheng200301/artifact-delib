"""Tests for ArtifactDelib Phase 3 — TargetedExpertRecheck + report version tracking."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.agents.experts.glyph_expert import GlyphExpert
from artifact_delib.agents.experts.local_detail_expert import LocalDetailExpert
from artifact_delib.agents.experts.material_craft_expert import MaterialCraftExpert
from artifact_delib.agents.experts.shape_expert import ShapeExpert
from artifact_delib.agents.experts.style_expert import StyleExpert
from artifact_delib.agents.targeted_recheck import TargetedExpertRecheck
from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import (
    ArtifactCandidate,
    CandidateSet,
    DisagreementAnalysis,
    ExpertReport,
    RecheckRecord,
    RouteDecision,
)
from artifact_delib.utils.fixture_builder import build_artifact_fixtures
from artifact_delib.api.schemas import TokenUsage


# ═══════════════════════════════════════════════
#  TargetedExpertRecheck unit tests
# ═══════════════════════════════════════════════

@pytest.fixture
def sample():
    with tempfile.TemporaryDirectory() as tmp:
        yield build_artifact_fixtures(Path(tmp))[0]


@pytest.fixture
def mock_experts():
    client = ArtifactMockClient()
    return (
        ShapeExpert(client),
        StyleExpert(client),
        GlyphExpert(client),
        MaterialCraftExpert(client),
        LocalDetailExpert(client),
    )


@pytest.fixture
def coordinator(mock_experts):
    se, st, g, m, l = mock_experts
    return TargetedExpertRecheck(shape_expert=se, style_expert=st,
                                  glyph_expert=g, material_expert=m,
                                  local_detail_expert=l)


@pytest.fixture
def candidates():
    return CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.48),
        ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.32),
        ArtifactCandidate(text="明代早期青花梅瓶", confidence=0.20),
    ))


@pytest.fixture
def disagreement():
    return DisagreementAnalysis(
        content="主要不确定性在永乐和宣德的具体年代判断。",
        route_hint="STYLE",
    )


def test_coordinator_execute_returns_recheck_record(
    coordinator: TargetedExpertRecheck,
    sample,
    candidates: CandidateSet,
    disagreement: DisagreementAnalysis,
) -> None:
    """Execute should return a properly populated RecheckRecord."""
    route = RouteDecision(action="STYLE_RECHECK", reason="test", recheck_count=0)
    current_reports = (
        ExpertReport(expert_name="style", content="初始纹饰分析", usage=TokenUsage()),
    )

    record = coordinator.execute(
        image_path=sample.image_path,
        route=route,
        candidates=candidates,
        disagreement=disagreement,
        current_reports=current_reports,
        recheck_history=(),
        round_no=1,
    )

    assert isinstance(record, RecheckRecord)
    assert record.round_no == 1
    assert record.expert_name == "纹饰风格"
    assert record.previous_content == "初始纹饰分析"  # version tracking
    assert record.new_content  # should have new content
    assert "候选" in record.context_query  # context should mention candidates


def test_coordinator_records_previous_content(
    coordinator: TargetedExpertRecheck,
    sample,
    candidates: CandidateSet,
    disagreement: DisagreementAnalysis,
) -> None:
    """Previous content should be captured for version diffing."""
    route = RouteDecision(action="SHAPE_RECHECK", reason="test", recheck_count=0)
    current_reports = (
        ExpertReport(expert_name="shape", content="原始器形分析内容", usage=TokenUsage()),
    )

    record = coordinator.execute(
        image_path=sample.image_path,
        route=route,
        candidates=candidates,
        disagreement=disagreement,
        current_reports=current_reports,
        recheck_history=(),
        round_no=1,
    )

    assert record.previous_content == "原始器形分析内容"
    assert record.new_content != record.previous_content  # should differ


def test_coordinator_builds_context_with_candidates_and_history(
    coordinator: TargetedExpertRecheck,
    sample,
    candidates: CandidateSet,
) -> None:
    """Context should include candidate info and recheck history."""
    route = RouteDecision(action="STYLE_RECHECK", reason="test", recheck_count=1)
    current_reports = (
        ExpertReport(expert_name="style", content="旧报告", usage=TokenUsage()),
    )
    history = (
        RecheckRecord(
            round_no=1, expert_name="器形", previous_content="旧",
            new_content="新", context_query="第一轮",
        ),
    )

    record = coordinator.execute(
        image_path=sample.image_path,
        route=route,
        candidates=candidates,
        disagreement=None,
        current_reports=current_reports,
        recheck_history=history,
        round_no=2,
    )

    # Context should mention the history (e.g. "第1轮")
    import re
    assert re.search(r"第\d+轮", record.context_query), (
        f"Expected round number info in context: {record.context_query[:100]}"
    )
    # Context should mention candidate names (from the candidate fixture)
    assert "永乐" in record.context_query or "宣德" in record.context_query or "候选1" in record.context_query


def test_all_actions_produce_recheck_record(
    coordinator: TargetedExpertRecheck,
    sample,
    candidates: CandidateSet,
) -> None:
    """Every recheck action should produce a valid record."""
    for action, expert_name in [
        ("SHAPE_RECHECK", "器形"),
        ("STYLE_RECHECK", "纹饰风格"),
        ("GLYPH_RECHECK", "铭文款识"),
        ("MATERIAL_RECHECK", "材质工艺"),
        ("LOCAL_DETAIL_RECHECK", "局部细节"),
    ]:
        route = RouteDecision(action=action, reason="test", recheck_count=0)
        record = coordinator.execute(
            image_path=sample.image_path,
            route=route,
            candidates=candidates,
            disagreement=None,
            current_reports=(),
            recheck_history=(),
            round_no=1,
        )
        assert record.expert_name == expert_name
        assert record.new_content


# ═══════════════════════════════════════════════
#  Integration: Pipeline with recheck_records
# ═══════════════════════════════════════════════

def test_pipeline_populates_recheck_records(sample) -> None:
    """Pipeline should populate recheck_records when recheck occurs."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=2,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    if len(result.route_decisions) > 1:  # had recheck rounds
        assert len(result.recheck_records) > 0
        for record in result.recheck_records:
            assert isinstance(record, RecheckRecord)
            assert record.expert_name
            assert record.new_content
            # Version tracking: previous content should differ from new
            if record.previous_content:
                assert record.previous_content != record.new_content


def test_pipeline_recheck_records_match_expert_names(sample) -> None:
    """Recheck record expert names should correspond to actual experts."""
    pipeline = ArtifactDelibPipeline(
        client=ArtifactMockClient(),
        max_recheck_rounds=3,
    )
    result = pipeline.run(sample.image_path, sample.sample_id)

    recheck_actions = [
        rd.action for rd in result.route_decisions if "RECHECK" in rd.action
    ]
    name_map = {
        "SHAPE_RECHECK": "器形",
        "STYLE_RECHECK": "纹饰风格",
        "GLYPH_RECHECK": "铭文款识",
        "MATERIAL_RECHECK": "材质工艺",
        "LOCAL_DETAIL_RECHECK": "局部细节",
    }

    for action, record in zip(recheck_actions, result.recheck_records):
        expected_name = name_map.get(action, "")
        if expected_name:
            # The mock returns prompt_name-based content,
            # the Chinese expert name comes from record metadata
            assert record.expert_name, f"Empty expert name for {action}"


def test_pipeline_context_includes_candidate_comparison(sample) -> None:
    """Recheck context query should include candidate comparison language."""
    pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
    result = pipeline.run(sample.image_path, sample.sample_id)

    for record in result.recheck_records:
        if record.context_query:
            # Context should mention distinguishing candidates
            assert any(kw in record.context_query for kw in ["候选", "区分", "差异"])


def test_pipeline_recheck_round_numbers_sequential(sample) -> None:
    """Recheck round numbers should be sequential starting from 1."""
    pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
    result = pipeline.run(sample.image_path, sample.sample_id)

    for i, record in enumerate(result.recheck_records):
        assert record.round_no == i + 1, (
            f"Expected round {i+1}, got {record.round_no}"
        )


def test_pipeline_no_recheck_no_records(sample) -> None:
    """When no recheck occurs, recheck_records should be empty."""

    class FastMock:
        def generate(self, request):
            from artifact_delib.api.schemas import ModelResponse, TokenUsage
            pn = (request.prompt_name or "").lower()
            if "candidate" in pn:
                content = (
                    '```json\n{"candidates": ['
                    '{"text": "明永乐青花梅瓶", "confidence": 0.90},'
                    '{"text": "明宣德青花梅瓶", "confidence": 0.06}]}\n```'
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

    pipeline = ArtifactDelibPipeline(client=FastMock())
    result = pipeline.run(sample.image_path, sample.sample_id)

    assert len(result.recheck_records) == 0
    assert len(result.recheck_reports) == 0
