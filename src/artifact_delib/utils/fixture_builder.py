"""Deterministic artifact fixture builder — synthetic images for testing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from artifact_delib.schemas import ArtifactSample

_FIXTURE_SAMPLES = [
    {
        "sample_id": "artifact-vase-001",
        "category": "瓷器",
        "fine_grained_type": "青花梅瓶",
        "period": "明永乐",
        "dynasty": "明",
        "material": "瓷",
        "craft": "青花",
        "region": "景德镇",
        "description": "BLUE VASE",
    },
    {
        "sample_id": "artifact-vase-002",
        "category": "瓷器",
        "fine_grained_type": "青花梅瓶",
        "period": "明宣德",
        "dynasty": "明",
        "material": "瓷",
        "craft": "青花",
        "region": "景德镇",
        "description": "BLUE VASE XUANDE",
    },
    {
        "sample_id": "artifact-bronze-001",
        "category": "青铜器",
        "fine_grained_type": "鼎",
        "period": "商代晚期",
        "dynasty": "商",
        "material": "青铜",
        "craft": "铸造",
        "region": "中原",
        "description": "BRONZE DING",
    },
    {
        "sample_id": "artifact-jade-001",
        "category": "玉器",
        "fine_grained_type": "玉璧",
        "period": "战国",
        "dynasty": "战国",
        "material": "玉",
        "craft": "雕刻",
        "region": "中原",
        "description": "JADE BI",
    },
]


def build_artifact_fixtures(root: Path) -> list[ArtifactSample]:
    """Create synthetic artifact images for testing the pipeline."""
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[ArtifactSample] = []

    for entry in _FIXTURE_SAMPLES:
        path = image_dir / f"{entry['sample_id']}.png"
        image = Image.new("RGB", (320, 320), color=(200, 190, 180))
        draw = ImageDraw.Draw(image)
        # Draw a simple artifact silhouette
        cx, cy = 160, 160
        # Vase body
        draw.ellipse((cx - 60, cy - 40, cx + 60, cy + 80), outline=(80, 70, 60), width=3)
        # Neck
        draw.rectangle((cx - 20, cy - 90, cx + 20, cy - 40), outline=(80, 70, 60), width=3)
        # Foot
        draw.rectangle((cx - 40, cy + 80, cx + 40, cy + 100), outline=(80, 70, 60), width=3)
        # Decoration
        draw.text((cx - 50, cy), entry["description"], fill=(60, 90, 140))
        image.save(path)

        samples.append(
            ArtifactSample(
                sample_id=entry["sample_id"],
                image_path=path,
                category=entry["category"],
                fine_grained_type=entry["fine_grained_type"],
                period=entry["period"],
                material=entry["material"],
                craft=entry["craft"],
                dynasty=entry["dynasty"],
                region=entry["region"],
                split="fixture",
                artifact_group_id=entry["sample_id"],
            )
        )
    return samples
