"""Metrics computed exclusively from structured predictions and known labels."""

from __future__ import annotations

from statistics import median

from pydantic import BaseModel, ConfigDict

from histodelib.schemas import Label, Prediction


class MetricsReport(BaseModel):
    """Metric output; unavailable comparisons remain ``None`` (rendered as N/A)."""

    model_config = ConfigDict(frozen=True)

    accuracy: float | None
    macro_f1: float | None
    miscaptioned_f1: float | None
    average_tokens: float | None = None
    token_saving: float | None = None
    correction_rate: float | None = None
    harm_rate: float | None = None
    per_class: dict[str, dict[str, float]] | None = None
    confusion_matrix: dict[str, dict[str, int]] | None = None
    balanced_accuracy: float | None = None
    coverage: float | None = None
    abstention_rate: float | None = None
    invalid_output_rate: float | None = None
    route_rate: float | None = None
    reinspection_rate: float | None = None
    judge_revision_rate: float | None = None
    correction_to_harm_ratio: float | None = None
    median_tokens: float | None = None
    p95_tokens: float | None = None


def compute_metrics(
    predictions: list[Prediction],
    labels: dict[str, Label],
    baseline_predictions: list[Prediction] | None = None,
) -> MetricsReport:
    """Compute task metrics for predictions whose reference labels are provided."""

    prediction_by_id = {prediction.sample_id: prediction for prediction in predictions}
    paired_records = [
        (prediction, labels[prediction.sample_id])
        for sample_id, prediction in prediction_by_id.items()
        if sample_id in labels
    ]
    paired = [(prediction.final_label, actual) for prediction, actual in paired_records]
    if not paired:
        return MetricsReport(accuracy=None, macro_f1=None, miscaptioned_f1=None)
    accuracy = sum(predicted == actual for predicted, actual in paired) / len(paired)
    f1_by_label = {label: _f1(label, paired) for label in Label}
    average_tokens = sum(prediction.usage.total_tokens for prediction, _ in paired_records) / len(
        paired_records
    )
    baseline_by_id = {
        prediction.sample_id: prediction for prediction in (baseline_predictions or [])
    }
    baseline_paired = [
        baseline_by_id[prediction.sample_id]
        for prediction, _ in paired_records
        if prediction.sample_id in baseline_by_id
    ]
    baseline_average = (
        sum(prediction.usage.total_tokens for prediction in baseline_paired) / len(baseline_paired)
        if baseline_paired
        else None
    )
    incorrect_initial = sum(
        prediction.initial_label is not None and prediction.initial_label != actual
        for prediction, actual in paired_records
    )
    corrected = sum(
        prediction.initial_label is not None
        and prediction.initial_label != actual
        and prediction.final_label == actual
        for prediction, actual in paired_records
    )
    correct_initial = sum(
        prediction.initial_label == actual
        for prediction, actual in paired_records
    )
    harmful = sum(
        prediction.initial_label == actual and prediction.final_label != actual
        for prediction, actual in paired_records
    )
    token_values = [prediction.usage.total_tokens for prediction, _ in paired_records]
    final_non_null = sum(prediction.final_label is not None for prediction, _ in paired_records)
    abstained = sum(prediction.final_label is None for prediction, _ in paired_records)
    invalid = sum(prediction.status == "INVALID_OUTPUT" for prediction, _ in paired_records)
    routes = sum(bool(prediction.evidence.get("route")) for prediction, _ in paired_records)
    reinspections = sum(
        bool(prediction.evidence.get("reinspection_api_calls", 0))
        for prediction, _ in paired_records
    )
    revisions = sum(
        prediction.evidence.get("judge_decision") == "REVISE"
        for prediction, _ in paired_records
    )
    per_class = {
        label.value: {
            "precision": _precision(label, paired),
            "recall": _recall(label, paired),
            "f1": f1_by_label[label],
        }
        for label in Label
    }
    confusion = {
        actual.value: {
            predicted.value: sum(
                prediction == predicted and actual_label == actual
                for prediction, actual_label in paired
            )
            for predicted in Label
        }
        for actual in Label
    }
    recalls = [_recall(label, paired) for label in Label]
    return MetricsReport(
        accuracy=accuracy,
        macro_f1=sum(f1_by_label.values()) / len(f1_by_label),
        miscaptioned_f1=f1_by_label[Label.MISCAPTIONED],
        average_tokens=average_tokens,
        token_saving=(1 - average_tokens / baseline_average) if baseline_average else None,
        correction_rate=(corrected / incorrect_initial) if incorrect_initial else None,
        harm_rate=(harmful / correct_initial) if correct_initial else None,
        per_class=per_class,
        confusion_matrix=confusion,
        balanced_accuracy=sum(recalls) / len(recalls),
        coverage=final_non_null / len(paired_records),
        abstention_rate=abstained / len(paired_records),
        invalid_output_rate=invalid / len(paired_records),
        route_rate=routes / len(paired_records),
        reinspection_rate=reinspections / len(paired_records),
        judge_revision_rate=revisions / len(paired_records),
        correction_to_harm_ratio=(corrected / harmful) if harmful else None,
        median_tokens=float(median(token_values)),
        p95_tokens=float(_percentile(token_values, 0.95)),
    )


def _f1(label: Label, paired: list[tuple[Label | None, Label]]) -> float:
    true_positive = sum(predicted == label and actual == label for predicted, actual in paired)
    false_positive = sum(predicted == label and actual != label for predicted, actual in paired)
    false_negative = sum(predicted != label and actual == label for predicted, actual in paired)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator


def _precision(label: Label, paired: list[tuple[Label | None, Label]]) -> float:
    true_positive = sum(predicted == label and actual == label for predicted, actual in paired)
    predicted_positive = sum(predicted == label for predicted, _ in paired)
    return true_positive / predicted_positive if predicted_positive else 0.0


def _recall(label: Label, paired: list[tuple[Label | None, Label]]) -> float:
    true_positive = sum(predicted == label and actual == label for predicted, actual in paired)
    actual_positive = sum(actual == label for _, actual in paired)
    return true_positive / actual_positive if actual_positive else 0.0


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])
