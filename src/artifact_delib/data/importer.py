"""Artifact dataset CSV importer — reads manifest CSV into ArtifactSample list."""

from __future__ import annotations

import csv
from pathlib import Path

from artifact_delib.schemas import ArtifactSample


REQUIRED_COLUMNS = {"sample_id", "image_path"}
OPTIONAL_COLUMNS = {
    "category", "fine_grained_type", "period", "dynasty",
    "material", "craft", "region", "artifact_group_id",
    "split", "source",
}


class ArtifactDatasetImporter:
    """Import artifact dataset from a CSV manifest file.

    The CSV must contain at least: sample_id, image_path
    Optional gold label columns: category, fine_grained_type, period, dynasty, etc.
    """

    def __init__(self, image_root: Path | None = None) -> None:
        self.image_root = image_root

    def import_manifest(self, manifest_path: Path) -> list[ArtifactSample]:
        """Read a CSV manifest and return validated ArtifactSample list."""
        root = (self.image_root or manifest_path.parent).resolve()
        samples: list[ArtifactSample] = []
        seen_ids: set[str] = set()

        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or ())

            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")

            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id:
                    raise ValueError("empty sample_id in row")
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                seen_ids.add(sample_id)

                # Resolve image path
                img_rel = (row.get("image_path") or "").strip()
                if not img_rel:
                    raise ValueError(f"empty image_path for {sample_id}")
                img_path = (root / img_rel).resolve()

                split_val = (row.get("split") or "").strip() or None
                if split_val not in {"train", "validation", "test", "fixture", "unassigned", None}:
                    split_val = None

                samples.append(ArtifactSample(
                    sample_id=sample_id,
                    image_path=img_path,
                    category=self._safe_str(row, "category"),
                    fine_grained_type=self._safe_str(row, "fine_grained_type"),
                    period=self._safe_str(row, "period"),
                    dynasty=self._safe_str(row, "dynasty"),
                    material=self._safe_str(row, "material"),
                    craft=self._safe_str(row, "craft"),
                    region=self._safe_str(row, "region"),
                    source=self._safe_str(row, "source"),
                    split=split_val,  # type: ignore[arg-type]
                    artifact_group_id=self._safe_str(row, "artifact_group_id"),
                ))

        return samples

    @staticmethod
    def _safe_str(row: dict[str, str], col: str) -> str | None:
        val = (row.get(col) or "").strip()
        return val if val else None
