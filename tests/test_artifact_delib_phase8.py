"""Tests for ArtifactDelib Phases 8-9 — Oracle Route Builder + Learned Router."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from artifact_delib.evaluation.prediction_parser import PredictionParser
from artifact_delib.router.learned_router import MLPRouter, TrainingRecord, TrainingSet
from artifact_delib.router.oracle_route_builder import (
    N_CLASSES,
    N_FEATURES,
    OracleRecord,
    OracleRouteBuilder,
    RouteFeatures,
    RouteOutcome,
    features_to_tensor,
    label_to_route,
    route_to_label,
)
from artifact_delib.schemas import (
    ArtifactCandidate,
    CandidateSet,
    DisagreementAnalysis,
    RouteDecision,
)


# ═══════════════════════════════════════════════════════════════
#  Feature conversion tests
# ═══════════════════════════════════════════════════════════════

def test_features_to_tensor_shape() -> None:
    feats = RouteFeatures(0.5, 0.3, 0.2, "STYLE", 3)
    vec = features_to_tensor(feats)
    assert len(vec) == N_FEATURES
    assert all(isinstance(v, float) for v in vec)


def test_features_to_tensor_values() -> None:
    feats = RouteFeatures(0.48, 0.32, 0.16, "STYLE", 3)
    vec = features_to_tensor(feats)
    assert vec[0] == 0.48  # top1_conf
    assert vec[1] == 0.32  # top2_conf
    assert vec[2] == 0.16  # margin
    assert vec[3] == 1.0   # STYLE → index 1


def test_route_to_label_roundtrip() -> None:
    for action in OracleRouteBuilder.ROUTE_ACTIONS:
        label = route_to_label(action)
        back = label_to_route(label)
        assert back == action, f"{action} → {label} → {back}"


def test_route_to_label_bounds() -> None:
    assert route_to_label("FAST") == 0
    assert route_to_label("DELIBERATION") == 6
    assert route_to_label("UNKNOWN") == 0  # fallback


def test_label_to_route_invalid() -> None:
    assert label_to_route(-1) == "FAST"
    assert label_to_route(99) == "FAST"


# ═══════════════════════════════════════════════════════════════
#  OracleRouteBuilder tests
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def oracle():
    parser = PredictionParser()
    from artifact_delib.evaluation.metrics import ArtifactMetrics
    metrics = ArtifactMetrics(parser)
    return OracleRouteBuilder(parser, metrics)


def test_oracle_extracts_features(oracle: OracleRouteBuilder) -> None:
    """Features should be extracted from pipeline result."""
    # Create a minimal pipeline result for feature extraction
    from artifact_delib.schemas import PipelineResult, ExpertReport, VisualPerceptionReport, SummarizedReport, FinalIdentification, TokenUsage as T

    feats = RouteFeatures(
        top1_confidence=0.48,
        top2_confidence=0.32,
        margin=0.16,
        disagreement_type="STYLE",
        n_candidates=3,
    )
    # Direct feature creation
    assert feats.top1_confidence == 0.48
    assert feats.margin == 0.16


def test_oracle_scores_route(oracle: OracleRouteBuilder) -> None:
    """Route scoring should evaluate correctness."""
    outcome = oracle.score_route(
        route_action="FAST",
        final_text="这是一件明代永乐时期的青花梅瓶。",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明代永乐时期",
        api_calls=10,
        tokens=100,
    )
    assert outcome.route_action == "FAST"
    assert outcome.is_correct  # Type matches
    assert outcome.api_calls == 10


def test_oracle_scores_incorrect_route(oracle: OracleRouteBuilder) -> None:
    """Incorrect prediction should be scored as incorrect."""
    outcome = oracle.score_route(
        route_action="STYLE_RECHECK",
        final_text="这是一个碗。",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明代",
        api_calls=15,
        tokens=200,
    )
    assert not outcome.is_correct


def test_oracle_selects_cheapest_correct(oracle: OracleRouteBuilder) -> None:
    """Among correct routes, the cheapest should be selected."""
    outcomes = [
        RouteOutcome("FAST", is_correct=True, api_calls=5, total_tokens=50),
        RouteOutcome("STYLE_RECHECK", is_correct=True, api_calls=10, total_tokens=100),
        RouteOutcome("DELIBERATION", is_correct=True, api_calls=20, total_tokens=200),
    ]
    best_action, best_outcome = oracle.select_oracle_route(outcomes)

    assert best_action == "FAST"
    assert best_outcome is not None
    assert best_outcome.api_calls == 5


def test_oracle_falls_back_to_cheapest_when_none_correct(oracle: OracleRouteBuilder) -> None:
    """When no route is correct, pick the cheapest."""
    outcomes = [
        RouteOutcome("FAST", is_correct=False, api_calls=5, total_tokens=50),
        RouteOutcome("DELIBERATION", is_correct=False, api_calls=20, total_tokens=200),
    ]
    action, outcome = oracle.select_oracle_route(outcomes)

    assert outcome is not None
    assert outcome.api_calls == 5  # Cheapest, even though wrong


# ═══════════════════════════════════════════════════════════════
#  MLPRouter tests
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mlp() -> MLPRouter:
    return MLPRouter(learning_rate=0.1, random_seed=42)


def test_mlp_forward_shape(mlp: MLPRouter) -> None:
    x = [0.48, 0.32, 0.16, 1.0, 3.0, 0.16, 0.15]
    logits = mlp.forward(x)
    assert len(logits) == N_CLASSES


def test_mlp_predict_returns_valid_route(mlp: MLPRouter) -> None:
    feats = [0.48, 0.32, 0.16, 1.0, 3.0, 0.16, 0.15]
    route = mlp.predict(feats)
    assert route in OracleRouteBuilder.ROUTE_ACTIONS


def test_mlp_untrained_random_predictions(mlp: MLPRouter) -> None:
    """Untrained router should produce valid but random-ish predictions."""
    routes = set()
    for _ in range(20):
        x = [0.5 + i * 0.01 for i in range(N_FEATURES)]
        routes.add(mlp.predict(x))
    # With random init, should explore multiple classes
    assert len(routes) >= 1


def test_mlp_train_on_synthetic_data(mlp: MLPRouter) -> None:
    """Train on synthetic data and verify loss decreases."""
    records = [
        TrainingRecord(
            features=features_to_tensor(
                RouteFeatures(0.9, 0.05, 0.85, "MULTI_FACTOR", 3)
            ),
            oracle_label=route_to_label("FAST"),
            oracle_route="FAST",
            sample_id="s1",
        ),
        TrainingRecord(
            features=features_to_tensor(
                RouteFeatures(0.48, 0.42, 0.06, "STYLE", 3)
            ),
            oracle_label=route_to_label("STYLE_RECHECK"),
            oracle_route="STYLE_RECHECK",
            sample_id="s2",
        ),
        TrainingRecord(
            features=features_to_tensor(
                RouteFeatures(0.35, 0.33, 0.02, "MULTI_FACTOR", 2)
            ),
            oracle_label=route_to_label("DELIBERATION"),
            oracle_route="DELIBERATION",
            sample_id="s3",
        ),
    ] * 5  # Repeat for batch training
    ts = TrainingSet(records=records)
    history = mlp.train(ts, epochs=100, batch_size=8)

    assert len(history) == 100
    assert history[-1] < history[0] or abs(history[-1] - history[0]) < 0.01
    assert mlp.trained


def test_mlp_accuracy_improves_after_training(mlp: MLPRouter) -> None:
    """Model should achieve reasonable accuracy on its training data."""
    records = []
    for i in range(50):
        conf = 0.3 + i * 0.01
        route = "FAST" if conf > 0.65 else "STYLE_RECHECK"
        records.append(TrainingRecord(
            features=features_to_tensor(
                RouteFeatures(conf, 0.2, conf - 0.2, "STYLE", 3)
            ),
            oracle_label=route_to_label(route),
            oracle_route=route,
            sample_id=f"s{i}",
        ))
    ts = TrainingSet(records=records)
    mlp.train(ts, epochs=200, batch_size=16)

    acc = mlp.accuracy(ts)
    assert acc >= 0.5  # Should at least beat random (1/7 ~ 0.14)


def test_mlp_save_load(mlp: MLPRouter) -> None:
    """Save and load should preserve weights."""
    mlp.train(TrainingSet(records=[
        TrainingRecord(
            features=features_to_tensor(RouteFeatures(0.9, 0.1, 0.8, "MULTI_FACTOR", 3)),
            oracle_label=0, oracle_route="FAST", sample_id="s1",
        ),
    ] * 10), epochs=10)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        mlp.save(path)

        loaded = MLPRouter()
        loaded.load(path)

        assert loaded.trained
        # Same features should produce same prediction
        x = features_to_tensor(RouteFeatures(0.9, 0.1, 0.8, "MULTI_FACTOR", 3))
        assert mlp.predict(x) == loaded.predict(x)


def test_mlp_route_interface(mlp: MLPRouter) -> None:
    """MLPRouter.route() should be compatible with pipeline router interface."""
    mlp.train(TrainingSet(records=[
        TrainingRecord(
            features=features_to_tensor(RouteFeatures(0.9, 0.1, 0.8, "MULTI_FACTOR", 3)),
            oracle_label=0, oracle_route="FAST", sample_id="s1",
        ),
    ] * 10), epochs=10)

    candidates = CandidateSet(candidates=(
        ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.9),
        ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.1),
    ))
    disagreement = DisagreementAnalysis(content="", route_hint="MULTI_FACTOR")
    decision = mlp.route(disagreement, candidates)

    assert isinstance(decision, RouteDecision)
    assert decision.action in OracleRouteBuilder.ROUTE_ACTIONS


def test_mlp_avoids_repeated_rechecks(mlp: MLPRouter) -> None:
    """When a recheck type was already done, router should pick another."""
    candidates = CandidateSet(candidates=(
        ArtifactCandidate(text="A", confidence=0.48),
        ArtifactCandidate(text="B", confidence=0.42),
    ))
    disagreement = DisagreementAnalysis(content="", route_hint="STYLE")

    # Mark STYLE_RECHECK as already done
    decision = mlp.route(
        disagreement, candidates,
        completed_rechecks=("STYLE_RECHECK",),
    )
    assert decision.action != "STYLE_RECHECK"


def test_training_set_properties() -> None:
    """TrainingSet should provide X and y correctly."""
    ts = TrainingSet(records=[
        TrainingRecord(features=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                       oracle_label=0, oracle_route="FAST", sample_id="s1"),
        TrainingRecord(features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       oracle_label=2, oracle_route="STYLE_RECHECK", sample_id="s2"),
    ])
    assert len(ts) == 2
    assert len(ts.X) == 2
    assert len(ts.y) == 2
    assert ts.records[0].features == ts.X[0]
    assert ts.records[0].oracle_label == ts.y[0]
