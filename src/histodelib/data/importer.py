"""Local CSV manifest importer; no network or dataset discovery."""

from __future__ import annotations

import csv
from pathlib import Path

from histodelib.schemas import Label, Sample

REQUIRED_COLUMNS = {"sample_id", "image_path", "caption", "label"}


def import_manifest(manifest_path: Path, image_root: Path) -> list[Sample]:
    """Import and validate a local CSV manifest with paths confined to ``image_root``."""

    root = image_root.resolve()
    samples: list[Sample] = []
    seen: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            image_path = (root / (row.get("image_path") or "")).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"image path outside image root: {row.get('image_path')}")
            samples.append(
                Sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    caption=row.get("caption", ""),
                    label=Label(row.get("label", "")),
                    original_group_id=row.get("original_group_id") or None,
                    source=row.get("source") or None,
                    license=row.get("license") or None,
                    domain=row.get("domain") or None,
                    conflict_type=row.get("conflict_type") or None,
                )
            )
    return samples
