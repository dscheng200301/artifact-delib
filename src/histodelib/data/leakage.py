"""Group-aware split leakage checks."""

from __future__ import annotations

from collections import defaultdict

from histodelib.schemas import Sample


def find_group_leakage(samples: list[Sample]) -> dict[str, set[str]]:
    """Return original-image groups assigned to more than one declared split."""

    split_by_group: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.original_group_id is not None and sample.split is not None:
            split_by_group[sample.original_group_id].add(sample.split)
    return {group: splits for group, splits in split_by_group.items() if len(splits) > 1}
