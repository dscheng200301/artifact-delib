"""Tests for ArtifactDelib Phase 7 — Ablation variants."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.ablations import (
    LegacyAblationFixedAllRecheck as AblationFixedAllRecheck,
    LegacyAblationFixedDeliberation as AblationFixedDeliberation,
    AblationFreeDebate,
    LegacyAblationNoDeferredJudge as AblationNoDeferredJudge,
    LegacyAblationNoDeliberation as AblationNoDeliberation,
    LegacyAblationNoMultiExpert as AblationNoMultiExpert,
    LegacyAblationNoRecheck as AblationNoRecheck,
    LegacyAblationNoRouter as AblationNoRouter,
    LegacyAblationSingleExpert as AblationSingleExpert,
)
from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.utils.fixture_builder import build_artifact_fixtures


@pytest.fixture
def sample():
    with tempfile.TemporaryDirectory() as tmp:
        yield build_artifact_fixtures(Path(tmp))[0]


@pytest.fixture
def client():
    return ArtifactMockClient()


# ═══════════════════════════════════════════════════════════════
#  A1: w/o Multi-Expert
# ═══════════════════════════════════════════════════════════════

def test_a1_runs_without_experts(sample, client) -> None:
    ablation = AblationNoMultiExpert(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert len(result.expert_reports) == 0  # No experts
    assert result.total_api_calls == 2  # VP + Judge only


def test_a1_fewer_calls_than_full(sample, client) -> None:
    full = ArtifactDelibPipeline(client)
    a1 = AblationNoMultiExpert(client)

    r_full = full.run(sample.image_path)
    r_a1 = a1.run(sample.image_path)

    assert r_a1.total_api_calls < r_full.total_api_calls


# ═══════════════════════════════════════════════════════════════
#  A2: Single General Expert
# ═══════════════════════════════════════════════════════════════

def test_a2_single_expert_runs(sample, client) -> None:
    ablation = AblationSingleExpert(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert len(result.expert_reports) == 1  # Only 1 generic expert
    assert result.final_identification.content


# ═══════════════════════════════════════════════════════════════
#  A3: w/o Dynamic Router
# ═══════════════════════════════════════════════════════════════

def test_a3_no_router_runs(sample, client) -> None:
    ablation = AblationNoRouter(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert len(result.route_decisions) == 0  # No routing
    assert len(result.recheck_reports) == 0  # No recheck
    assert len(result.expert_reports) == 5  # But experts still run


def test_a3_same_calls_as_b2(sample, client) -> None:
    from artifact_delib.baselines import FixedMultiExpertBaseline

    r_a3 = AblationNoRouter(client).run(sample.image_path)
    r_b2 = FixedMultiExpertBaseline(client).run(sample.image_path)

    assert r_a3.total_api_calls == r_b2.total_api_calls  # Both fixed path


# ═══════════════════════════════════════════════════════════════
#  A4: w/o Targeted Recheck
# ═══════════════════════════════════════════════════════════════

def test_a4_no_recheck_runs(sample, client) -> None:
    ablation = AblationNoRecheck(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert result.disagreement_analysis is not None  # Disagreement still runs
    assert len(result.recheck_reports) == 0  # But no recheck
    assert len(result.route_decisions) == 0  # No routing


# ═══════════════════════════════════════════════════════════════
#  A5: Fixed All Recheck
# ═══════════════════════════════════════════════════════════════

def test_a5_fixed_recheck_runs_all_5(sample, client) -> None:
    ablation = AblationFixedAllRecheck(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert len(result.recheck_reports) == 5  # All 5 rechecks run
    # 5 route decisions should be recheck actions
    recheck_routes = [r for r in result.route_decisions if "RECHECK" in r.action]
    assert len(recheck_routes) == 5


# ═══════════════════════════════════════════════════════════════
#  A8: w/o Deliberation
# ═══════════════════════════════════════════════════════════════

def test_a8_no_deliberation_runs(sample, client) -> None:
    ablation = AblationNoDeliberation(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert result.deliberation_result is None  # Deliberation stripped
    assert result.final_identification.content


# ═══════════════════════════════════════════════════════════════
#  A9: Free Debate
# ═══════════════════════════════════════════════════════════════

def test_a9_free_debate_runs(sample, client) -> None:
    ablation = AblationFreeDebate(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert result.final_identification.content


# ═══════════════════════════════════════════════════════════════
#  A10: Fixed Deliberation
# ═══════════════════════════════════════════════════════════════

def test_a10_fixed_deliberation_runs(sample, client) -> None:
    ablation = AblationFixedDeliberation(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    # Should have deliberation result (enforced)
    assert result.deliberation_result is not None


# ═══════════════════════════════════════════════════════════════
#  A12: w/o Deferred Judge
# ═══════════════════════════════════════════════════════════════

def test_a12_no_deferred_judge_runs(sample, client) -> None:
    ablation = AblationNoDeferredJudge(client)
    result = ablation.run(sample.image_path)

    assert result.status == "COMPLETED"
    assert len(result.initial_candidates.candidates) == 0  # No candidates generated
    assert result.final_identification.content  # Judge still produces output


# ═══════════════════════════════════════════════════════════════
#  Cross-ablation comparisons
# ═══════════════════════════════════════════════════════════════

def test_ablations_increasing_calls(sample, client) -> None:
    """Expected call order: A1 < A2 < A3 < A5 < Full"""
    r_a1 = AblationNoMultiExpert(client).run(sample.image_path)
    r_a2 = AblationSingleExpert(client).run(sample.image_path)
    r_a3 = AblationNoRouter(client).run(sample.image_path)
    r_a5 = AblationFixedAllRecheck(client).run(sample.image_path)
    r_full = ArtifactDelibPipeline(client).run(sample.image_path)

    assert r_a1.total_api_calls == 2
    assert r_a2.total_api_calls == 5  # VP + generic + summarizer + candidate + judge
    assert r_a3.total_api_calls == 9  # VP + 5 experts + summarizer + candidate + judge
    assert r_a5.total_api_calls > r_a3.total_api_calls  # Has rechecks
    assert r_full.total_api_calls > r_a1.total_api_calls


def test_all_ablations_produce_valid_output(sample, client) -> None:
    ablations = {
        "A1": AblationNoMultiExpert(client),
        "A2": AblationSingleExpert(client),
        "A3": AblationNoRouter(client),
        "A4": AblationNoRecheck(client),
        "A5": AblationFixedAllRecheck(client),
        "A8": AblationNoDeliberation(client),
        "A9": AblationFreeDebate(client),
        "A10": AblationFixedDeliberation(client),
        "A12": AblationNoDeferredJudge(client),
    }
    for name, ablation in ablations.items():
        result = ablation.run(sample.image_path, sample.sample_id)
        assert result.status == "COMPLETED", f"{name}: {result.status}"
        assert result.final_identification.content, f"{name}: empty output"
