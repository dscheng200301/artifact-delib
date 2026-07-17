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
    token_saving: float | None = None


def compute_metrics(predictions: list[Prediction], labels: dict[str, Label]) -> MetricsReport:
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
    return MetricsReport(
        accuracy=accuracy,
        macro_f1=sum(f1_by_label.values()) / len(f1_by_label),
        miscaptioned_f1=f1_by_label[Label.MISCAPTIONED],
    )


def _f1(label: Label, paired: list[tuple[Label | None, Label]]) -> float:
    true_positive = sum(predicted == label and actual == label for predicted, actual in paired)
    false_positive = sum(predicted == label and actual != label for predicted, actual in paired)
    false_negative = sum(predicted != label and actual == label for predicted, actual in paired)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator
