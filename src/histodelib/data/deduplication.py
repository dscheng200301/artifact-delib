"""Exact and perceptual image duplicate checks for local manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

from histodelib.schemas import Sample


def image_sha256(path: Path) -> str:
    """Hash image bytes without modifying or normalizing the source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_exact_duplicates(samples: list[Sample]) -> dict[str, tuple[str, ...]]:
    """Return byte-identical image groups containing more than one sample."""

    groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.image_path.exists() and sample.image_path.is_file():
            groups[image_sha256(sample.image_path)].append(sample.sample_id)
    return {digest: tuple(ids) for digest, ids in groups.items() if len(ids) > 1}


def find_perceptual_duplicates(
    samples: list[Sample], max_distance: int = 0
) -> dict[str, tuple[str, ...]]:
    """Group images with equal/near-equal perceptual hashes."""

    hashes: dict[str, imagehash.ImageHash] = {}
    for sample in samples:
        try:
            with Image.open(sample.image_path) as image:
                hashes[sample.sample_id] = imagehash.phash(image)
        except (OSError, ValueError):
            continue
    groups: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for sample_id, current in hashes.items():
        if sample_id in seen:
            continue
        matching = [
            other for other, candidate in hashes.items() if current - candidate <= max_distance
        ]
        if len(matching) > 1:
            key = str(current)
            groups[key].extend(matching)
            seen.update(matching)
    return {key: tuple(dict.fromkeys(ids)) for key, ids in groups.items()}
