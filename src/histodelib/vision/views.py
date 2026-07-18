"""Create auditable image views without local model inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

ViewTarget = Literal["full", "patch", "glyph", "panor"]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageView:
    """Bytes and provenance for one deterministic image view."""

    target: ViewTarget
    data: bytes
    source_sha256: str
    view_sha256: str
    bbox: BBox | None
    scale: float
    generation_version: str
    reason_code: str


def create_view(
    image_path: Path,
    target: ViewTarget,
    *,
    bbox: BBox | None = None,
    scale: float = 1.0,
) -> ImageView:
    """Create a PNG view and record the exact source/view provenance."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    source = image_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    with Image.open(BytesIO(source)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        resolved_bbox = bbox
        reason_code = "full_view"
        if target != "full":
            if bbox is None:
                resolved_bbox = None
                reason_code = "target_unavailable_full_fallback"
            else:
                _validate_bbox(bbox, width, height)
                image = image.crop(bbox)
                reason_code = f"{target}_bbox"
        if scale != 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            )
        output = BytesIO()
        image.save(output, format="PNG", optimize=False)
    data = output.getvalue()
    return ImageView(
        target=target,
        data=data,
        source_sha256=source_sha256,
        view_sha256=hashlib.sha256(data).hexdigest(),
        bbox=resolved_bbox,
        scale=scale,
        generation_version="views-v1",
        reason_code=reason_code,
    )


def _validate_bbox(bbox: BBox, width: int, height: int) -> None:
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox must be inside image bounds; got {bbox} for {(width, height)}")
