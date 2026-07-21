"""Tests for ArtifactDelib Phase 6 — Baselines (B1-B4)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.baselines import (
    DirectVLMBaseline,
    FixedFullBaseline,
    FixedMultiExpertBaseline,
    GenericMADBaseline,
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
#  B1: DirectVLM
# ═══════════════════════════════════════════════════════════════

def test_b1_direct_vlm_runs(sample, client) -> None:
    """B1 should complete with a single model call."""
    baseline = DirectVLMBaseline(client)
    result = baseline.run(sample.image_path, sample.sample_id)

    assert result.status == "COMPLETED"
    assert result.total_api_calls == 1
    assert result.final_identification.content


def test_b1_direct_vlm_has_no_experts(sample, client) -> None:
    """B1 should have no expert reports (single call only)."""
    baseline = DirectVLMBaseline(client)
    result = baseline.run(sample.image_path)

    assert len(result.expert_reports) == 0
    assert len(result.route_decisions) == 0
    assert len(result.recheck_reports) == 0


# ═══════════════════════════════════════════════════════════════
#  B2: FixedMultiExpert
# ═══════════════════════════════════════════════════════════════

def test_b2_fixed_multi_expert_runs(sample, client) -> None:
    """B2 should run VP + 5 experts + summarizer + judge."""
    baseline = FixedMultiExpertBaseline(client)
    result = baseline.run(sample.image_path, sample.sample_id)

    assert result.status == "COMPLETED"
    # VP(1) + 5 experts(5) + summarizer(1) + candidate(1) + judge(1) = 9
    assert result.total_api_calls == 9
    assert len(result.expert_reports) == 5
    assert result.final_identification.content


def test_b2_fixed_multi_expert_no_routing(sample, client) -> None:
    """B2 should have NO route decisions (no router used)."""
    baseline = FixedMultiExpertBaseline(client)
    result = baseline.run(sample.image_path)

    assert len(result.route_decisions) == 0
    assert len(result.recheck_reports) == 0
    assert result.deliberation_result is None


def test_b2_calls_match_fixed_schedule(sample, client) -> None:
    """B2 should make exactly 9 calls for every sample."""
    baseline = FixedMultiExpertBaseline(client)
    r1 = baseline.run(sample.image_path, "s1")
    r2 = baseline.run(sample.image_path, "s2")

    assert r1.total_api_calls == 9
    assert r2.total_api_calls == 9  # Always same


# ═══════════════════════════════════════════════════════════════
#  B3: GenericMAD
# ═══════════════════════════════════════════════════════════════

def test_b3_generic_mad_runs(sample, client) -> None:
    """B3 should complete with N agents + debate rounds + judge."""
    baseline = GenericMADBaseline(client, n_agents=4, n_rounds=2)
    result = baseline.run(sample.image_path, sample.sample_id)

    assert result.status == "COMPLETED"
    # Initial: 4 agents(4) + Debate rounds: 4 agents × (2-1) = 4 + Judge(1) = 9
    assert result.total_api_calls == 4 + 4 + 1  # = 9
    assert result.final_identification.content


def test_b3_generic_mad_varying_agents(sample, client) -> None:
    """B3 with different N should produce different call counts."""
    r2 = GenericMADBaseline(client, n_agents=2, n_rounds=2).run(
        sample.image_path, "s2",
    )
    r4 = GenericMADBaseline(client, n_agents=4, n_rounds=2).run(
        sample.image_path, "s4",
    )
    assert r2.total_api_calls == 2 + 2 + 1  # = 5
    assert r4.total_api_calls == 4 + 4 + 1  # = 9


# ═══════════════════════════════════════════════════════════════
#  B4: FixedFull
# ═══════════════════════════════════════════════════════════════

def test_b4_fixed_full_runs(sample, client) -> None:
    """B4 should run everything unconditionally."""
    baseline = FixedFullBaseline(client)
    result = baseline.run(sample.image_path, sample.sample_id)

    assert result.status == "COMPLETED"
    # VP(1) + 5 experts(5) + summarizer(1) + candidate(1) + disagreement(1)
    # + 5 rechecks(5) + re-summarize(1) + deliberation(≈3) + judge(1)
    assert result.total_api_calls > 10 + 5  # At least initial + all rechecks
    assert len(result.recheck_reports) >= 5  # All 5 rechecks done
    assert result.deliberation_result is not None


def test_b4_always_does_all_rechecks(sample, client) -> None:
    """B4 should do all 5 recheck types in every run."""
    baseline = FixedFullBaseline(client)
    r1 = baseline.run(sample.image_path, "s1")
    r2 = baseline.run(sample.image_path, "s2")

    # Both should have same number of recheck reports (all 5)
    assert len(r1.recheck_reports) == 5
    assert len(r2.recheck_reports) == 5


def test_b4_always_does_deliberation(sample, client) -> None:
    """B4 should always run deliberation."""
    baseline = FixedFullBaseline(client)
    result = baseline.run(sample.image_path)

    assert result.deliberation_result is not None
    assert len(result.deliberation_result.rounds) >= 1


# ═══════════════════════════════════════════════════════════════
#  Cross-baseline comparison
# ═══════════════════════════════════════════════════════════════

def test_b1_calls_fewest(sample, client) -> None:
    """B1 (DirectVLM) should have the fewest API calls."""
    r_b1 = DirectVLMBaseline(client).run(sample.image_path, "s")
    r_b2 = FixedMultiExpertBaseline(client).run(sample.image_path, "s")
    r_b4 = FixedFullBaseline(client).run(sample.image_path, "s")
    r_b5 = ArtifactDelibPipeline(client).run(sample.image_path, "s")

    assert r_b1.total_api_calls < r_b2.total_api_calls  # 1 < 9
    assert r_b2.total_api_calls < r_b4.total_api_calls  # 9 < 14+
    # B5 may or may not be less than B4 depending on routing
    assert r_b5.total_api_calls <= r_b4.total_api_calls + 5  # B5 ≈ B2 + routing


def test_all_baselines_produce_valid_output(sample, client) -> None:
    """Every baseline should produce a non-empty final identification."""
    baselines = [
        ("B1", DirectVLMBaseline(client)),
        ("B2", FixedMultiExpertBaseline(client)),
        ("B3", GenericMADBaseline(client, n_agents=2, n_rounds=2)),
        ("B4", FixedFullBaseline(client)),
        ("B5", ArtifactDelibPipeline(client)),
    ]
    for name, bl in baselines:
        result = bl.run(sample.image_path, sample.sample_id)
        assert result.status == "COMPLETED", f"{name} failed: {result.status}"
        assert result.final_identification.content, f"{name}: empty final output"
        assert result.total_api_calls > 0, f"{name}: zero API calls"


def test_baseline_api_calls_increasing(sample, client) -> None:
    """Expected call order: B1 < B2 < B3 < B4 (B3=X, B5=vary)."""
    r_b1 = DirectVLMBaseline(client).run(sample.image_path)
    r_b2 = FixedMultiExpertBaseline(client).run(sample.image_path)
    r_b4 = FixedFullBaseline(client).run(sample.image_path)

    assert r_b1.total_api_calls == 1
    assert r_b2.total_api_calls == 9
    assert r_b4.total_api_calls > r_b2.total_api_calls
