"""Group-aware statistical helpers for future authorized experiments."""

from __future__ import annotations

import random
from collections import defaultdict


def group_bootstrap_mean(
    values: list[float],
    groups: list[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return mean and a group-resampled percentile interval."""

    if len(values) != len(groups) or not values:
        raise ValueError("values and groups must be non-empty and have equal length")
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[group].append(float(value))
    group_values = list(grouped.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sampled = [rng.choice(group_values) for _ in group_values]
        flattened = [value for group in sampled for value in group]
        estimates.append(sum(flattened) / len(flattened))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return (
        sum(values) / len(values),
        estimates[max(0, int(alpha * n_resamples))],
        estimates[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))],
    )


def paired_accuracy_delta(
    predictions_a: dict[str, bool], predictions_b: dict[str, bool]
) -> float:
    """Compute paired correctness delta only on the shared sample IDs."""

    ids = sorted(set(predictions_a) & set(predictions_b))
    if not ids:
        raise ValueError("no shared sample IDs")
    return sum(predictions_a[sample_id] - predictions_b[sample_id] for sample_id in ids) / len(ids)
