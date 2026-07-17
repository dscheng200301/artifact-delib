"""Deterministic group-aware train/validation/test splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from histodelib.schemas import Sample


def split_samples(
    samples: list[Sample], seed: int = 0, train_ratio: float = 0.8, validation_ratio: float = 0.1
) -> dict[str, list[Sample]]:
    """Assign whole original-image groups to deterministic split buckets."""

    if (
        not 0 < train_ratio < 1
        or not 0 <= validation_ratio < 1
        or train_ratio + validation_ratio > 1
    ):
        raise ValueError("split ratios must be valid and sum to at most one")
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.original_group_id or sample.sample_id].append(sample)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    total = len(group_ids)
    train_count = max(1, round(total * train_ratio)) if total else 0
    validation_count = round(total * validation_ratio)
    assignments: dict[str, str] = {}
    for index, group_id in enumerate(group_ids):
        assignments[group_id] = (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
    result: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    for group_id, members in groups.items():
        result[assignments[group_id]].extend(
            member.model_copy(update={"split": assignments[group_id]}) for member in members
        )
    return result
