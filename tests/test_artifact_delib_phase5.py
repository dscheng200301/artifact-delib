"""Tests for ArtifactDelib Phase 5 — Evaluation Parser, Metrics, Experiment Logger."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.evaluation.experiment_logger import ExperimentLogger, ExperimentRecord
from artifact_delib.evaluation.metrics import ArtifactMetrics, SampleEvaluation
from artifact_delib.evaluation.prediction_parser import ParsedIdentification, PredictionParser


# ═══════════════════════════════════════════════
#  PredictionParser tests
# ═══════════════════════════════════════════════

@pytest.fixture
def parser():
    return PredictionParser()


# ── Category extraction ──

@pytest.mark.parametrize("text,expected", [
    ("这是一件瓷器，青花梅瓶。", "瓷器"),
    ("该器物是一件青铜器，器形为鼎。", "青铜器"),
    ("综合判断为玉器，可能是玉璧。", "玉器"),
])
def test_parser_extracts_category(parser: PredictionParser, text: str, expected: str) -> None:
    result = parser.parse(text)
    assert result.category == expected, f"{text!r} → expected {expected!r}, got {result.category!r}"


# ── Fine-grained type extraction ──

@pytest.mark.parametrize("text,expected", [
    ("该器物是一件明代早期青花梅瓶。", "梅瓶"),
    ("器形为青铜鼎，属于商代晚期。", "鼎"),
    ("是一件玉璧，战国时期。", "玉璧"),
])
def test_parser_extracts_type(parser: PredictionParser, text: str, expected: str) -> None:
    result = parser.parse(text)
    assert result.fine_grained_type == expected, (
        f"{text!r} → expected {expected!r}, got {result.fine_grained_type!r}"
    )


# ── Period extraction ──

@pytest.mark.parametrize("text,expected_period,expected_dynasty", [
    ("该器物更可能是一件明代永乐时期的青花梅瓶。", "明代永乐时期", "明"),
    ("属于商代晚期青铜器。", "商代晚期", "商"),
    ("推测为战国时期玉璧。", "战国时期", "战国"),
    ("综合判断为明宣德青花梅瓶。", "明宣德", "明"),
])
def test_parser_extracts_period(
    parser: PredictionParser,
    text: str,
    expected_period: str,
    expected_dynasty: str,
) -> None:
    result = parser.parse(text)
    assert result.period == expected_period, (
        f"{text!r} → expected {expected_period!r}, got {result.period!r}"
    )
    assert result.dynasty == expected_dynasty, (
        f"{text!r} → expected dynasty {expected_dynasty!r}, got {result.dynasty!r}"
    )


def test_parser_handles_unknown_text(parser: PredictionParser) -> None:
    """Should not crash on text with no recognizable keywords."""
    result = parser.parse("这是一段没有文物识别关键词的文本。")
    assert result.category is None
    assert result.fine_grained_type is None
    assert result.period is None


def test_parser_handles_empty_text(parser: PredictionParser) -> None:
    """Should handle empty input gracefully."""
    result = parser.parse("")
    assert result.category is None
    assert result.fine_grained_type is None


def test_parser_raw_text_truncated(parser: PredictionParser) -> None:
    """Raw text should be truncated to 200 chars."""
    long_text = "器" * 500 + "这是一件明永乐青花梅瓶"
    result = parser.parse(long_text)
    assert len(result.raw_text) <= 200


# ── Type disambiguation ──

def test_parser_longest_type_match(parser: PredictionParser) -> None:
    """Should return the most specific (longest) type keyword."""
    text = "该器物既不是瓶也不是梅瓶也不是玉壶春瓶，而是一件典型的玉壶春瓶。"
    result = parser.parse(text)
    assert result.fine_grained_type == "玉壶春瓶"


# ═══════════════════════════════════════════════
#  ArtifactMetrics tests
# ═══════════════════════════════════════════════

@pytest.fixture
def metrics():
    return ArtifactMetrics()


def test_metrics_evaluate_sample_correct(
    metrics: ArtifactMetrics,
) -> None:
    """Sample where prediction matches gold."""
    r = metrics.evaluate_sample(
        sample_id="test-001",
        final_text="这是一件明代永乐时期的青花梅瓶。",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明代",
    )
    assert r.category_correct
    # Type matches: both "梅瓶"
    assert r.type_correct


def test_metrics_evaluate_sample_incorrect(
    metrics: ArtifactMetrics,
) -> None:
    """Sample where prediction does NOT match gold."""
    r = metrics.evaluate_sample(
        sample_id="test-002",
        final_text="这是一件茶壶。",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明永乐",
    )
    assert not r.type_correct
    assert not r.period_correct


def test_metrics_compute_aggregate(
    metrics: ArtifactMetrics,
) -> None:
    """Should compute aggregate metrics from sample evaluations."""
    evals = [
        SampleEvaluation("s1", category_correct=True, type_correct=True, period_correct=True, joint_correct=True),
        SampleEvaluation("s2", category_correct=True, type_correct=True, period_correct=False, joint_correct=False),
        SampleEvaluation("s3", category_correct=True, type_correct=False, period_correct=False, joint_correct=False),
        SampleEvaluation("s4", category_correct=True, type_correct=True, period_correct=True, joint_correct=True),
    ]
    result = metrics.compute_metrics(evals, token_counts=[10, 20, 30, 40])

    assert result.n_samples == 4
    assert result.category_accuracy == 1.0
    assert result.type_accuracy == 0.75
    assert result.period_accuracy == 0.50
    assert result.joint_accuracy == 0.50
    assert result.average_tokens == 25.0


def test_metrics_empty_evaluations(
    metrics: ArtifactMetrics,
) -> None:
    """Should handle empty evaluation list."""
    result = metrics.compute_metrics([])
    assert result.n_samples == 0
    assert result.type_accuracy is None


def test_metrics_correction_detection(
    metrics: ArtifactMetrics,
) -> None:
    """Should detect correction when initial wrong → final correct."""
    r = metrics.evaluate_sample(
        sample_id="s1",
        final_text="明永乐青花梅瓶",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明永乐",
        initial_text="这是一件陶器碗",
    )
    # Initial parse: "陶器" category, "碗" type → wrong
    # Final parse: "瓷器" category, "梅瓶" type, "明永乐" period → correct
    assert r.type_correct
    assert r.corrected  # was wrong, now correct


# ═══════════════════════════════════════════════
#  ExperimentLogger tests
# ═══════════════════════════════════════════════

def test_logger_start_and_complete_experiment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(Path(tmp))
        record = logger.start_experiment(
            experiment_id="exp-001",
            method="artifact_delib_rule",
            config={"max_recheck_rounds": 2},
        )
        assert record.status == "PENDING"
        assert record.method == "artifact_delib_rule"

        completed = logger.complete_experiment(
            metrics={"accuracy": 0.85},
            n_samples=100,
        )
        assert completed.status == "COMPLETED"
        assert completed.n_samples == 100
        assert completed.metrics["accuracy"] == 0.85


def test_logger_persists_to_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(Path(tmp))
        logger.start_experiment("exp-001", "direct_vlm")
        logger.complete_experiment({"acc": 0.75}, 50)

        loaded = logger.load_history()
        assert len(loaded) >= 1
        assert loaded[0].experiment_id == "exp-001"


def test_logger_no_history_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(Path(tmp) / "nonexistent")
        records = logger.load_history()
        assert records == []


def test_logger_save_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(Path(tmp))
        path = logger.save_predictions("exp-001", [
            {"sample_id": "s1", "prediction": "明永乐青花梅瓶"},
            {"sample_id": "s2", "prediction": "商代青铜鼎"},
        ])
        assert path.exists()


# ═══════════════════════════════════════════════
#  Top-K Accuracy tests (academic FGVR standard)
# ═══════════════════════════════════════════════

def test_top1_alias_matches_top1_field(metrics: ArtifactMetrics) -> None:
    """Backward-compat alias should equal the top1_* field."""
    evals = [
        SampleEvaluation("s1", category_correct=True, type_correct=True,
                          period_correct=True, joint_correct=True),
    ]
    result = metrics.compute_metrics(evals)
    assert result.top1_type_accuracy == result.type_accuracy
    assert result.top1_category_accuracy == result.category_accuracy
    assert result.top1_period_accuracy == result.period_accuracy
    assert result.top1_joint_accuracy == result.joint_accuracy


def test_evaluate_sample_top5_type(
    metrics: ArtifactMetrics,
) -> None:
    """Top-5 type accuracy: gold should match if any of top-5 candidates parses to gold."""
    r = metrics.evaluate_sample(
        sample_id="s1",
        final_text="明永乐青花碗",             # final: "碗" (wrong)
        gold_category="瓷器",
        gold_type="梅瓶",                      # gold: "梅瓶"
        gold_period="明永乐",
        candidate_texts=[
            "明永乐青花碗",                    # candidate 1: 碗 (wrong)
            "明永乐青花盘",                    # candidate 2: 盘 (wrong)
            "明永乐青花梅瓶",                  # candidate 3: 梅瓶 ← MATCH
            "明宣德青花梅瓶",                  # candidate 4
            "清康熙青花梅瓶",                  # candidate 5
        ],
    )
    # Final prediction: "碗" → wrong
    assert not r.type_correct
    # But gold "梅瓶" appears in top-3 candidate texts → top3 hit
    assert r.type_in_top3
    assert r.type_in_top5


def test_evaluate_sample_top5_miss(
    metrics: ArtifactMetrics,
) -> None:
    """Top-5 miss when gold doesn't appear in any candidate."""
    r = metrics.evaluate_sample(
        sample_id="s1",
        final_text="明永乐青花碗",
        gold_category="瓷器",
        gold_type="梅瓶",
        gold_period="明永乐",
        candidate_texts=[
            "明永乐青花碗",
            "明永乐青花盘",
            "明永乐青花壶",
            "明永乐青花罐",
            "明永乐青花杯",
        ],
    )
    assert not r.type_in_top3
    assert not r.type_in_top5


def test_compute_top5_accuracy(metrics: ArtifactMetrics) -> None:
    """Aggregate Top-1 / Top-3 / Top-5 accuracy from mixed evaluations."""
    evals = [
        # Sample 1: top1 correct + in top3/5
        SampleEvaluation("s1", type_correct=True, type_in_top3=True, type_in_top5=True,
                         gold_type="梅瓶"),
        # Sample 2: top1 wrong but in top3
        SampleEvaluation("s2", type_correct=False, type_in_top3=True, type_in_top5=True,
                         gold_type="梅瓶"),
        # Sample 3: only in top5 (missed top3)
        SampleEvaluation("s3", type_correct=False, type_in_top3=False, type_in_top5=True,
                         gold_type="梅瓶"),
        # Sample 4: missed everything
        SampleEvaluation("s4", type_correct=False, type_in_top3=False, type_in_top5=False,
                         gold_type="梅瓶"),
    ]
    result = metrics.compute_metrics(evals)
    assert result.top1_type_accuracy == 0.25    # 1/4
    assert result.top3_type_accuracy == 0.50    # 2/4
    assert result.top5_type_accuracy == 0.75    # 3/4


# ═══════════════════════════════════════════════
#  Per-class P/R/F1 tests
# ═══════════════════════════════════════════════

def test_per_class_precision_recall_f1(metrics: ArtifactMetrics) -> None:
    """Per-class P/R/F1 should compute correctly for multi-class predictions."""
    evals = [
        # 2 瓷器 samples: 1 TP, 0 FP → P=1.0, R=1.0
        SampleEvaluation("s1", predicted_category="瓷器", gold_category="瓷器"),
        SampleEvaluation("s2", predicted_category="瓷器", gold_category="瓷器"),
        # 1 玉器 sample correctly predicted
        SampleEvaluation("s3", predicted_category="玉器", gold_category="玉器"),
        # 1 玉器 sample misclassified as 瓷器 → 瓷器 gets +1 FP, 玉器 gets +1 FN
        SampleEvaluation("s4", predicted_category="瓷器", gold_category="玉器"),
    ]
    result = metrics.compute_metrics(evals)
    assert result.per_category is not None

    cat_ciqi = result.per_category["瓷器"]
    # 瓷器: TP=2, FP=1, FN=0 → P=2/3, R=2/2=1.0, F1=4/5=0.8
    assert abs(cat_ciqi.precision - 2 / 3) < 1e-6
    assert abs(cat_ciqi.recall - 1.0) < 1e-6
    assert abs(cat_ciqi.f1 - 0.8) < 1e-6
    assert cat_ciqi.support == 2

    cat_yuqi = result.per_category["玉器"]
    # 玉器: TP=1, FP=0, FN=1 → P=1.0, R=1/2=0.5, F1=2/3≈0.667
    assert abs(cat_yuqi.precision - 1.0) < 1e-6
    assert abs(cat_yuqi.recall - 0.5) < 1e-6
    assert cat_yuqi.support == 2

    # Macro F1 = (0.8 + 0.667) / 2 ≈ 0.733
    assert result.macro_f1_category is not None
    expected_macro = (cat_ciqi.f1 + cat_yuqi.f1) / 2
    assert abs(result.macro_f1_category - expected_macro) < 1e-6
