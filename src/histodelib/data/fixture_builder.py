"""Build deterministic images used only for offline workflow verification."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from histodelib.schemas import Label, Sample

_MARKERS = {"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"}


def build_fixture(root: Path) -> list[Sample]:
    """Create one deliberately simple image per task label beneath ``root``."""

    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("fixture-true-1", Label.TRUE, "Synthetic 1912 harbor photograph", "1912 HARBOR"),
        ("fixture-true-2", Label.TRUE, "Synthetic 1920 harbor photograph", "1920 HARBOR"),
        ("fixture-true-3", Label.TRUE, "Synthetic 1930 station photograph", "1930 STATION"),
        ("fixture-true-4", Label.TRUE, "Synthetic 1940 square photograph", "1940 SQUARE"),
        (
            "fixture-miscaptioned-1",
            Label.MISCAPTIONED,
            "Synthetic 1913 harbor photograph",
            "1912 HARBOR",
        ),
        (
            "fixture-miscaptioned-2",
            Label.MISCAPTIONED,
            "Synthetic 1921 harbor photograph",
            "1920 HARBOR",
        ),
        (
            "fixture-miscaptioned-3",
            Label.MISCAPTIONED,
            "Synthetic 1931 station photograph",
            "1930 STATION",
        ),
        (
            "fixture-miscaptioned-4",
            Label.MISCAPTIONED,
            "Synthetic 1941 square photograph",
            "1940 SQUARE",
        ),
        (
            "fixture-ooc-1",
            Label.OUT_OF_CONTEXT,
            "Synthetic mountain expedition photograph",
            "1912 HARBOR",
        ),
        (
            "fixture-ooc-2",
            Label.OUT_OF_CONTEXT,
            "Synthetic mountain camp photograph",
            "1920 HARBOR",
        ),
        (
            "fixture-ooc-3",
            Label.OUT_OF_CONTEXT,
            "Synthetic mountain trail photograph",
            "1930 STATION",
        ),
        (
            "fixture-ooc-4",
            Label.OUT_OF_CONTEXT,
            "Synthetic mountain ridge photograph",
            "1940 SQUARE",
        ),
    )
    samples: list[Sample] = []
    for sample_id, label, caption, inscription in definitions:
        path = image_dir / f"{sample_id}.png"
        image = Image.new("L", (320, 180), color=215)
        draw = ImageDraw.Draw(image)
        draw.rectangle((14, 14, 306, 166), outline=55, width=3)
        draw.text((72, 82), inscription, fill=45)
        image.save(path)
        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=path,
                caption=caption,
                label=label,
                original_group_id=sample_id,
                source="synthetic fixture builder",
                split="fixture",
                fixture_markers=set(_MARKERS),
            )
        )
    return samples
