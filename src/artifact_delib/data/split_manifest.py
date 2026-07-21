"""Split manifest — load split assignments and inject them into ArtifactSample objects.

Split files are plain text files with one sample_id per line.
The directory structure is::

    splits/
    ├── train.txt
    ├── validation.txt
    └── test.txt
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.schemas import ArtifactSample


class SplitManifest:
    """Load and apply split assignments from split files.

    Usage::

        split_manifest = SplitManifest.load("data/artifact/splits")
        samples = split_manifest.assign(all_samples)
    """

    def __init__(
        self,
        train_ids: set[str],
        validation_ids: set[str],
        test_ids: set[str],
    ) -> None:
        self._train_ids = train_ids
        self._validation_ids = validation_ids
        self._test_ids = test_ids

        # Validate: no sample ID can be in multiple splits
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError if any ID appears in more than one split."""
        train_val = self._train_ids & self._validation_ids
        train_test = self._train_ids & self._test_ids
        val_test = self._validation_ids & self._test_ids
        overlaps = train_val | train_test | val_test
        if overlaps:
            raise ValueError(
                f"Sample IDs appear in multiple splits: {sorted(overlaps)}"
            )

    @classmethod
    def load(cls, split_dir: str | Path) -> SplitManifest:
        """Load split files from a directory or a single file.

        If split_dir is a directory, looks for train.txt, validation.txt, test.txt.
        If split_dir is a single file, uses it as the test split only.
        """
        split_path = Path(split_dir)

        if split_path.is_file():
            # Single file mode: use as test split
            test_ids = _load_ids(split_path)
            return cls(
                train_ids=set(),
                validation_ids=set(),
                test_ids=test_ids,
            )

        # Directory mode
        train_path = split_path / "train.txt"
        validation_path = split_path / "validation.txt"
        test_path = split_path / "test.txt"

        train_ids = _load_ids(train_path) if train_path.exists() else set()
        validation_ids = _load_ids(validation_path) if validation_path.exists() else set()
        test_ids = _load_ids(test_path) if test_path.exists() else set()

        return cls(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )

    def assign(
        self,
        samples: list[ArtifactSample],
    ) -> list[ArtifactSample]:
        """Assign split labels to samples based on their IDs.

        Returns a new list with updated split fields.
        Samples not found in any split get split=None.
        """
        result: list[ArtifactSample] = []
        for s in samples:
            split: str | None = None
            if s.sample_id in self._train_ids:
                split = "train"
            elif s.sample_id in self._validation_ids:
                split = "validation"
            elif s.sample_id in self._test_ids:
                split = "test"

            result.append(
                ArtifactSample(
                    sample_id=s.sample_id,
                    image_path=s.image_path,
                    category=s.category,
                    fine_grained_type=s.fine_grained_type,
                    period=s.period,
                    material=s.material,
                    craft=s.craft,
                    dynasty=s.dynasty,
                    region=s.region,
                    source=s.source,
                    split=split,
                    artifact_group_id=s.artifact_group_id,
                )
            )
        return result

    @property
    def train_ids(self) -> set[str]:
        return self._train_ids

    @property
    def validation_ids(self) -> set[str]:
        return self._validation_ids

    @property
    def test_ids(self) -> set[str]:
        return self._test_ids

    def summary(self) -> str:
        parts = []
        if self._train_ids:
            parts.append(f"train={len(self._train_ids)}")
        if self._validation_ids:
            parts.append(f"validation={len(self._validation_ids)}")
        if self._test_ids:
            parts.append(f"test={len(self._test_ids)}")
        return f"SplitManifest({', '.join(parts)})"


def _load_ids(path: Path) -> set[str]:
    """Load a set of sample IDs from a text file (one per line)."""
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }