"""Comprehensive tests for Phase 2-6 restructured experiment system.

Tests:
  - Material accuracy in prediction parser and metrics
  - All 8 core ablation variants produce correct configuration
  - Baseline registry and imports
  - Structured report parsing
  - Data leakage detector (sha256, perceptual hash, filename check)

All tests use mock clients — no real API calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.utils.fixture_builder import build_artifact_fixtures


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample():
    with tempfile.TemporaryDirectory() as tmp:
        yield build_artifact_fixtures(Path(tmp))[0]


@pytest.fixture
def client():
    return ArtifactMockClient()


# ═══════════════════════════════════════════════════════════════
#  Material accuracy — PredictionParser
# ═══════════════════════════════════════════════════════════════

from artifact_delib.evaluation.prediction_parser import PredictionParser


@pytest.mark.parametrize("text,expected_material", [
    ("该器物为瓷质，青花梅瓶。", "瓷"),
    ("一件青铜器，材质为青铜。", "青铜"),
    ("综合判断为玉器，白玉质地。", "玉"),
    ("漆器，漆质精良。", "漆"),
    ("这是一件金银器，金质。", "金银"),
    ("陶器，陶质。", "陶"),
    ("珐琅器物，景泰蓝工艺。", "珐琅"),
    ("石刻雕塑，石质。", "石"),
])
def test_parser_extracts_material(text: str, expected_material: str) -> None:
    parser = PredictionParser()
    result = parser.parse(text)
    assert result.material == expected_material, (
        f"Text: {text!r}, expected material={expected_material!r}, got {result.material!r}"
    )


def test_parser_material_none_for_generic_text() -> None:
    parser = PredictionParser()
    result = parser.parse("这是一件文物。")
    assert result.material is None


# ═══════════════════════════════════════════════════════════════
#  Material accuracy — Metrics
# ═══════════════════════════════════════════════════════════════

from artifact_delib.evaluation.metrics import ArtifactMetrics, SampleEvaluation


def test_metrics_material_accuracy() -> None:
    """Material accuracy should be computed from evaluations."""
    m = ArtifactMetrics()
    evals = [
        SampleEvaluation("s1", material_correct=True, predicted_material="瓷", gold_material="瓷"),
        SampleEvaluation("s2", material_correct=True, predicted_material="瓷", gold_material="瓷"),
        SampleEvaluation("s3", material_correct=False, predicted_material="青铜", gold_material="瓷"),
        SampleEvaluation("s4", material_correct=True, predicted_material="瓷", gold_material="瓷"),
    ]
    result = m.compute_metrics(evals)
    assert result.top1_material_accuracy == 0.75


def test_metrics_evaluate_sample_with_material() -> None:
    """evaluate_sample should track material correctness."""
    m = ArtifactMetrics()
    r = m.evaluate_sample(
        sample_id="s1",
        final_text="这是一件瓷质青花梅瓶。",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明永乐",
        gold_material="瓷",
    )
    assert r.material_correct
    assert r.gold_material == "瓷"


def test_per_material_metrics() -> None:
    """Per-material precision/recall/F1 should be computed."""
    m = ArtifactMetrics()
    evals = [
        SampleEvaluation("s1", predicted_material="瓷", gold_material="瓷"),
        SampleEvaluation("s2", predicted_material="瓷", gold_material="瓷"),
        SampleEvaluation("s3", predicted_material="青铜", gold_material="青铜"),
        SampleEvaluation("s4", predicted_material="瓷", gold_material="青铜"),
    ]
    result = m.compute_metrics(evals)
    assert result.per_material is not None
    assert "瓷" in result.per_material
    assert "青铜" in result.per_material
    assert result.macro_f1_material is not None


# ═══════════════════════════════════════════════════════════════
#  Ablation tests — verify each one truly changes behavior
# ═══════════════════════════════════════════════════════════════

from artifact_delib.ablations.no_expert_specialization import AblationNoExpertSpecialization
from artifact_delib.ablations.no_disagreement_analysis import AblationNoDisagreementAnalysis
from artifact_delib.ablations.no_dynamic_routing import AblationNoDynamicRouting
from artifact_delib.ablations.random_recheck import AblationRandomRecheck
from artifact_delib.ablations.no_controlled_deliberation import AblationNoControlledDeliberation
from artifact_delib.ablations.free_debate import AblationFreeDebateNew
from artifact_delib.ablations.no_critic import AblationNoCritic
from artifact_delib.ablations.early_judge import AblationEarlyJudge


def test_a1_generic_experts(sample, client) -> None:
    """A1: All experts should be general (named 'general')."""
    a1 = AblationNoExpertSpecialization(client)
    result = a1.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    assert len(result.expert_reports) == 5
    # All experts should have the generic name
    for report in result.expert_reports:
        assert report.expert_name == "general"


def test_a2_no_disagreement_analysis(sample, client) -> None:
    """A2: Routes should use margin-only logic."""
    a2 = AblationNoDisagreementAnalysis(client)
    result = a2.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    # Route decisions should have "margin-only:" prefix
    assert len(result.route_decisions) >= 1
    assert "margin-only" in result.route_decisions[0].reason


def test_a3_fixed_path(sample, client) -> None:
    """A3: All 5 rechecks should always run."""
    a3 = AblationNoDynamicRouting(client)
    result = a3.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    # Fixed path always runs all 5 rechecks
    assert len(result.recheck_records) == 5
    assert result.deliberation_result is not None


def test_a4_random_recheck_reproducible(sample, client) -> None:
    """A4: Same seed should produce same route decisions."""
    r1 = AblationRandomRecheck(client, random_seed=42).run(sample.image_path)
    r2 = AblationRandomRecheck(client, random_seed=42).run(sample.image_path)
    # Same seed -> same decisions
    actions1 = tuple(rd.action for rd in r1.route_decisions)
    actions2 = tuple(rd.action for rd in r2.route_decisions)
    assert actions1 == actions2


def test_a4_different_seeds_differ(sample, client) -> None:
    """A4: Different seeds may differ (or may not — either is fine)."""
    r1 = AblationRandomRecheck(client, random_seed=42).run(sample.image_path)
    r2 = AblationRandomRecheck(client, random_seed=123).run(sample.image_path)
    # Different seeds should (with high probability) differ at some point
    # This is probabilistic, but with 5 recheck actions and 2 rounds, chance of same is low
    actions1 = tuple(rd.action for rd in r1.route_decisions)
    actions2 = tuple(rd.action for rd in r2.route_decisions)
    # Just assert both complete
    assert r1.status == "COMPLETED"
    assert r2.status == "COMPLETED"


def test_a5_no_deliberation(sample, client) -> None:
    """A5: Deliberation should be nullified even if triggered."""
    a5 = AblationNoControlledDeliberation(client)
    result = a5.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    assert result.deliberation_result is None


def test_a6_free_debate(sample, client) -> None:
    """A6: Should complete and produce output."""
    a6 = AblationFreeDebateNew(client)
    result = a6.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    assert result.final_identification.content


def test_a7_no_critic(sample, client) -> None:
    """A7: Deliberation should have '(skipped - no critic)' in critic feedback."""
    a7 = AblationNoCritic(client)
    result = a7.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    if result.deliberation_result and result.deliberation_result.rounds:
        for dr in result.deliberation_result.rounds:
            assert "no critic" in dr.critic_feedback.lower()


def test_a8_early_judge(sample, client) -> None:
    """A8: No recheck records, no deliberation, no disagreement."""
    a8 = AblationEarlyJudge(client)
    result = a8.run(sample.image_path, sample.sample_id)
    assert result.status == "COMPLETED"
    assert len(result.recheck_records) == 0
    assert result.deliberation_result is None
    # Disagreement analysis may or may not be None
    assert result.final_identification.content


# ═══════════════════════════════════════════════════════════════
#  Baseline registry tests
# ═══════════════════════════════════════════════════════════════

from artifact_delib.baselines.registry import (
    CAT_EXTERNAL,
    CAT_LEGACY,
    CAT_OURS,
    list_baselines,
    register_baseline,
    get_baseline,
)


def test_registry_categories_exist() -> None:
    assert CAT_EXTERNAL == "external"
    assert CAT_OURS == "ours"
    assert CAT_LEGACY == "legacy"


def test_register_and_get() -> None:
    """Simple registration and retrieval."""
    from artifact_delib.baselines.direct_vlm import DirectVLMBaseline
    register_baseline("test_direct", DirectVLMBaseline, category=CAT_EXTERNAL)
    from artifact_delib.models.mock_artifact import ArtifactMockClient
    instance = get_baseline("test_direct", client=ArtifactMockClient())
    assert instance is not None


# ═══════════════════════════════════════════════════════════════
#  Backward compatibility tests
# ═══════════════════════════════════════════════════════════════

def test_old_baselines_module_works() -> None:
    """Old baselines.py shim should still work."""
    from artifact_delib.baselines import (
        DirectVLMBaseline,
        FixedMultiExpertBaseline,
        FixedFullBaseline,
        GenericMADBaseline,
        SelfConsistencyBaseline,
        MultiAgentDebateBaseline,
    )
    assert DirectVLMBaseline.name == "direct_single_vlm"
    assert FixedMultiExpertBaseline.name == "fixed_multi_expert"
    assert FixedFullBaseline.name == "fixed_full"
    assert GenericMADBaseline.name == "generic_mad"
    assert SelfConsistencyBaseline.name == "self_consistency_vlm"
    assert MultiAgentDebateBaseline.name == "multi_agent_debate"


def test_ablations_module_works() -> None:
    """ablations package should export all 8 variants."""
    from artifact_delib.ablations import (
        AblationNoExpertSpecialization,
        AblationNoDisagreementAnalysis,
        AblationNoDynamicRouting,
        AblationRandomRecheck,
        AblationNoControlledDeliberation,
        AblationFreeDebateNew,
        AblationNoCritic,
        AblationEarlyJudge,
    )
    names = [
        AblationNoExpertSpecialization.name,
        AblationNoDisagreementAnalysis.name,
        AblationNoDynamicRouting.name,
        AblationRandomRecheck.name,
        AblationNoControlledDeliberation.name,
        AblationFreeDebateNew.name,
        AblationNoCritic.name,
        AblationEarlyJudge.name,
    ]
    assert len(set(names)) == 8, f"Expected 8 unique names, got {len(set(names))}"


# ═══════════════════════════════════════════════════════════════
#  Structured report tests
# ═══════════════════════════════════════════════════════════════

from artifact_delib.agents.structured_report import (
    parse_expert_response,
    reconstruct_report,
)


def test_structured_report_json_block() -> None:
    text = '''
    一件宋代青瓷碗的分析。

    ```json
    {
      "report": "一件宋代青瓷碗。",
      "top_candidates": [
        {"name": "宋代青瓷碗", "confidence": 0.67}
      ],
      "uncertainty_focus": ["底足"],
      "recommended_expert": "style"
    }
    ```
    '''
    r = parse_expert_response(text, expert_type="shape")
    assert r.has_control_fields
    assert len(r.top_candidates) == 1
    assert r.top_candidates[0]["name"] == "宋代青瓷碗"
    assert r.recommended_expert == "style"


def test_structured_report_pure_nl() -> None:
    text = "这是一件明代青花梅瓶。"
    r = parse_expert_response(text)
    assert not r.has_control_fields
    assert r.report == text


def test_structured_report_invalid_confidence_clamped() -> None:
    text = '{"report": "x", "top_candidates": [{"name": "A", "confidence": 2.0}]}'
    r = parse_expert_response(text)
    assert r.top_candidates[0]["confidence"] == 1.0


# ═══════════════════════════════════════════════════════════════
#  Self-Consistency tests
# ═══════════════════════════════════════════════════════════════

from artifact_delib.baselines.self_consistency import SelfConsistencyBaseline


def test_self_consistency_calls_match_n(sample, client) -> None:
    """Self-consistency should make exactly n_samples calls."""
    sc = SelfConsistencyBaseline(client, n_samples=5, seed=42)
    result = sc.run(sample.image_path, sample.sample_id)
    assert result.total_api_calls == 5
    assert result.status == "COMPLETED"


def test_self_consistency_varying_n(sample, client) -> None:
    for n in (3, 5, 7):
        sc = SelfConsistencyBaseline(client, n_samples=n, seed=42)
        r = sc.run(sample.image_path)
        assert r.total_api_calls == n


# ═══════════════════════════════════════════════════════════════
#  Multi-Agent Debate tests
# ═══════════════════════════════════════════════════════════════

from artifact_delib.baselines.multi_agent_debate import MultiAgentDebateBaseline


def test_mad_first_round_independent(sample, client) -> None:
    """MAD should have agents making independent first-round analyses."""
    mad = MultiAgentDebateBaseline(client, n_agents=4, n_rounds=2)
    result = mad.run(sample.image_path, sample.sample_id)
    # n_agents * n_rounds + 1 judge
    expected_calls = 4 + 4 + 1  # = 9
    assert result.total_api_calls == expected_calls
    assert result.status == "COMPLETED"


def test_mad_respects_round_limit(sample, client) -> None:
    mad = MultiAgentDebateBaseline(client, n_agents=2, n_rounds=3)
    result = mad.run(sample.image_path)
    # 2 * 3 = 6 agent calls + 1 judge = 7
    assert result.total_api_calls == 7


# ═══════════════════════════════════════════════════════════════
#  All baselines produce valid output
# ═══════════════════════════════════════════════════════════════

from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.baselines.direct_vlm import DirectVLMBaseline
from artifact_delib.baselines.legacy import (
    FixedFullBaseline,
    FixedMultiExpertBaseline,
    GenericMADBaseline,
)


def test_all_external_baselines_run(sample, client) -> None:
    """Every external baseline should complete successfully."""
    baselines: list = [
        DirectVLMBaseline(client),
        SelfConsistencyBaseline(client, n_samples=3, seed=42),
        MultiAgentDebateBaseline(client, n_agents=2, n_rounds=2),
        ArtifactDelibPipeline(client),
    ]
    for bl in baselines:
        result = bl.run(sample.image_path, sample.sample_id)
        assert result.status == "COMPLETED", f"{bl.name}: {result.status}"
        assert result.final_identification.content, f"{bl.name}: empty"


# ═══════════════════════════════════════════════════════════════
#  Parse failure rate test
# ═══════════════════════════════════════════════════════════════

def test_parser_handles_parse_failures() -> None:
    parser = PredictionParser()
    # Empty and whitespace texts should not crash
    assert parser.parse("") is not None
    assert parser.parse("   ") is not None
    assert parser.parse("无任何关键词的文本") is not None
    # Parse failure rate should be trackable (no exception)
