"""Group-aware artifact dataset splitter — 70/10/20 by artifact_group_id."""

from __future__ import annotations

import random
from collections import defaultdict

from artifact_delib.schemas import ArtifactSample


class ArtifactDatasetSplitter:
    """Split artifact samples into train/validation/test by artifact_group_id.

    This prevents data leakage: all images of the same artifact stay in the same split.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        if not (0 < train_ratio < 1 and 0 <= validation_ratio < 1
                and train_ratio + validation_ratio <= 1):
            raise ValueError("invalid split ratios")
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed

    def split(
        self, samples: list[ArtifactSample]
    ) -> dict[str, list[ArtifactSample]]:
        """Split samples and return dict with keys: train, validation, test."""
        # Group by artifact_group_id
        groups: dict[str, list[ArtifactSample]] = defaultdict(list)
        for s in samples:
            gid = s.artifact_group_id or s.sample_id
            groups[gid].append(s)

        # Shuffle group IDs
        group_ids = list(groups)
        rng = random.Random(self.seed)
        rng.shuffle(group_ids)

        # Assign splits
        n_total = len(group_ids)
        n_train = max(1, round(n_total * self.train_ratio))
        n_val = max(0, round(n_total * self.validation_ratio))

        result: dict[str, list[ArtifactSample]] = {
            "train": [], "validation": [], "test": [],
        }
        for i, gid in enumerate(group_ids):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            for s in groups[gid]:
                result[split].append(ArtifactSample(
                    sample_id=s.sample_id, image_path=s.image_path,
                    category=s.category, fine_grained_type=s.fine_grained_type,
                    period=s.period, dynasty=s.dynasty,
                    material=s.material, craft=s.craft, region=s.region,
                    source=s.source, split=split,  # type: ignore[arg-type]
                    artifact_group_id=s.artifact_group_id,
                ))

        return result

    def summary(self, split_result: dict[str, list[ArtifactSample]]) -> str:
        """Return a human-readable split summary."""
        lines = ["Dataset Split Summary:", "=" * 40]
        for split_name in ["train", "validation", "test"]:
            samples = split_result.get(split_name, [])
            unique_objects = len(set(
                s.artifact_group_id for s in samples
            ))
            lines.append(
                f"  {split_name:12s}: {len(samples):4d} samples, "
                f"{unique_objects:4d} unique objects"
            )
        return "\n".join(lines)
