"""Metrics computed exclusively from structured predictions and known labels."""

from __future__ import annotations

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


def compute_metrics(
    predictions: list[Prediction],
    labels: dict[str, Label],
    baseline_predictions: list[Prediction] | None = None,
) -> MetricsReport:
    """Compute task metrics for predictions whose reference labels are provided."""

    paired = [
        (prediction.final_label, labels[prediction.sample_id])
        for prediction in predictions
        if prediction.sample_id in labels
    ]
    if not paired:
        return MetricsReport(accuracy=None, macro_f1=None, miscaptioned_f1=None)
    accuracy = sum(predicted == actual for predicted, actual in paired) / len(paired)
    f1_by_label = {label: _f1(label, paired) for label in Label}
    average_tokens = sum(
        prediction.usage.total_tokens for prediction, _ in zip(predictions, paired, strict=False)
    ) / len(paired)
    baseline_average = (
        sum(prediction.usage.total_tokens for prediction in baseline_predictions)
        / len(baseline_predictions)
        if baseline_predictions
        else None
    )
    incorrect_initial = sum(
        prediction.initial_label is not None and prediction.initial_label != actual
        for prediction, (_, actual) in zip(predictions, paired, strict=False)
    )
    corrected = sum(
        prediction.initial_label is not None
        and prediction.initial_label != actual
        and prediction.final_label == actual
        for prediction, (_, actual) in zip(predictions, paired, strict=False)
    )
    correct_initial = sum(
        prediction.initial_label == actual
        for prediction, (_, actual) in zip(predictions, paired, strict=False)
    )
    harmful = sum(
        prediction.initial_label == actual and prediction.final_label != actual
        for prediction, (_, actual) in zip(predictions, paired, strict=False)
    )
    return MetricsReport(
        accuracy=accuracy,
        macro_f1=sum(f1_by_label.values()) / len(f1_by_label),
        miscaptioned_f1=f1_by_label[Label.MISCAPTIONED],
        average_tokens=average_tokens,
        token_saving=(1 - average_tokens / baseline_average) if baseline_average else None,
        correction_rate=(corrected / incorrect_initial) if incorrect_initial else None,
        harm_rate=(harmful / correct_initial) if correct_initial else None,
    )


def _f1(label: Label, paired: list[tuple[Label | None, Label]]) -> float:
    true_positive = sum(predicted == label and actual == label for predicted, actual in paired)
    false_positive = sum(predicted == label and actual != label for predicted, actual in paired)
    false_negative = sum(predicted != label and actual == label for predicted, actual in paired)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator
