"""Group-aware split leakage checks."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from histodelib.schemas import Sample


def find_group_leakage(samples: list[Sample]) -> dict[str, set[str]]:
    """Return original-image groups assigned to more than one declared split."""

    split_by_group: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.original_group_id is not None and sample.split is not None:
            split_by_group[sample.original_group_id].add(sample.split)
        if sample.split is not None and sample.image_path.exists() and sample.image_path.is_file():
            digest = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
            split_by_group[f"image_sha256:{digest}"].add(sample.split)
        if sample.split is not None:
            normalized_caption = " ".join(sample.caption.lower().split())
            split_by_group[f"caption:{normalized_caption}"].add(sample.split)
    return {group: splits for group, splits in split_by_group.items() if len(splits) > 1}
