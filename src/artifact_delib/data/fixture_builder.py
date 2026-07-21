"""Comprehensive fixture builder — diverse synthetic artifact samples for testing.

Generates 24+ samples across 4 categories with varying difficulty levels,
suitable for testing the full pipeline, routing, and evaluation.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from artifact_delib.schemas import ArtifactSample

# 24 fixture samples covering 4 categories × varying difficulty
_FIXTURES = [
    # ── Ceramics / 瓷器 (6 samples, 2 artifact groups with 3 views each) ──
    ("fixture-vase-001-a", "瓷器", "青花梅瓶", "明永乐", "明", "瓷", "青花", "景德镇", "g-vase-001",
     "VASE YONGLE", (0, 60, 200)),
    ("fixture-vase-001-b", "瓷器", "青花梅瓶", "明永乐", "明", "瓷", "青花", "景德镇", "g-vase-001",
     "VASE YONGLE SIDE", (60, 100, 200)),
    ("fixture-vase-001-c", "瓷器", "青花梅瓶", "明永乐", "明", "瓷", "青花", "景德镇", "g-vase-001",
     "VASE YONGLE BASE", (30, 90, 220)),

    ("fixture-vase-002-a", "瓷器", "青花梅瓶", "明宣德", "明", "瓷", "青花", "景德镇", "g-vase-002",
     "VASE XUANDE", (0, 60, 200)),
    ("fixture-vase-002-b", "瓷器", "青花梅瓶", "明宣德", "明", "瓷", "青花", "景德镇", "g-vase-002",
     "VASE XUANDE SIDE", (60, 100, 200)),
    ("fixture-vase-003-a", "瓷器", "青花碗", "明成化", "明", "瓷", "青花", "景德镇", "g-bowl-001",
     "BOWL CHENGHUA", (0, 40, 180)),

    # ── Bronzes / 青铜器 (6 samples) ──
    ("fixture-bronze-001-a", "青铜器", "鼎", "商代晚期", "商", "青铜", "铸造", "中原", "g-ding-001",
     "BRONZE DING", (80, 20, 100)),
    ("fixture-bronze-001-b", "青铜器", "鼎", "商代晚期", "商", "青铜", "铸造", "中原", "g-ding-001",
     "BRONZE DING SIDE", (60, 40, 120)),
    ("fixture-bronze-002-a", "青铜器", "觚", "商代", "商", "青铜", "铸造", "中原", "g-gu-001",
     "BRONZE GU", (80, 20, 100)),
    ("fixture-bronze-003-a", "青铜器", "爵", "西周早期", "西周", "青铜", "铸造", "中原", "g-jue-001",
     "BRONZE JUE", (80, 20, 100)),
    ("fixture-bronze-003-b", "青铜器", "爵", "西周早期", "西周", "青铜", "铸造", "中原", "g-jue-001",
     "BRONZE JUE SIDE", (60, 40, 120)),
    ("fixture-bronze-004-a", "青铜器", "尊", "西周", "西周", "青铜", "铸造", "中原", "g-zun-001",
     "BRONZE ZUN", (80, 20, 100)),

    # ── Jade / 玉器 (6 samples) ──
    ("fixture-jade-001-a", "玉器", "玉璧", "战国", "战国", "玉", "雕刻", "中原", "g-bi-001",
     "JADE BI", (0, 80, 140)),
    ("fixture-jade-001-b", "玉器", "玉璧", "战国", "战国", "玉", "雕刻", "中原", "g-bi-001",
     "JADE BI SIDE", (20, 100, 160)),
    ("fixture-jade-002-a", "玉器", "玉琮", "良渚文化", "良渚", "玉", "雕刻", "江南", "g-cong-001",
     "JADE CONG", (0, 80, 140)),
    ("fixture-jade-003-a", "玉器", "玉璋", "商代", "商", "玉", "雕刻", "中原", "g-zhang-001",
     "JADE ZHANG", (0, 80, 140)),
    ("fixture-jade-003-b", "玉器", "玉璋", "商代", "商", "玉", "雕刻", "中原", "g-zhang-001",
     "JADE ZHANG SIDE", (20, 100, 160)),
    ("fixture-jade-004-a", "玉器", "玉圭", "周代", "周", "玉", "雕刻", "中原", "g-gui-001",
     "JADE GUI", (0, 80, 140)),

    # ── Lacquer / 漆器 (6 samples) ──
    ("fixture-lacquer-001-a", "漆器", "漆盒", "汉代", "汉", "漆", "髹漆", "楚地", "g-box-001",
     "LACQUER BOX", (40, 50, 150)),
    ("fixture-lacquer-001-b", "漆器", "漆盒", "汉代", "汉", "漆", "髹漆", "楚地", "g-box-001",
     "LACQUER BOX TOP", (20, 30, 130)),
    ("fixture-lacquer-002-a", "漆器", "漆盘", "战国", "战国", "漆", "髹漆", "楚地", "g-dish-001",
     "LACQUER DISH", (40, 50, 150)),
    ("fixture-lacquer-003-a", "漆器", "漆耳杯", "西汉", "西汉", "漆", "髹漆", "楚地", "g-cup-001",
     "LACQUER CUP", (40, 50, 150)),
    ("fixture-lacquer-003-b", "漆器", "漆耳杯", "西汉", "西汉", "漆", "髹漆", "楚地", "g-cup-001",
     "LACQUER CUP SIDE", (20, 30, 130)),
    ("fixture-lacquer-004-a", "漆器", "漆奁", "宋代", "宋", "漆", "髹漆", "江南", "g-lian-001",
     "LACQUER LIAN", (40, 50, 150)),
]


def build_comprehensive_fixtures(root: Path) -> list[ArtifactSample]:
    """Build 24 diverse artifact samples for testing.

    Returns:
        list[ArtifactSample] ready for splitting and evaluation.
    """
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[ArtifactSample] = []

    for entry in _FIXTURES:
        (sid, cat, ftype, period, dynasty, mat, craft, region, gid, desc,
         color) = entry
        path = image_dir / f"{sid}.png"

        _draw_artifact(path, desc, color)
        img_size = path.stat().st_size

        samples.append(ArtifactSample(
            sample_id=sid,
            image_path=path,
            category=cat,
            fine_grained_type=ftype,
            period=period,
            dynasty=dynasty,
            material=mat,
            craft=craft,
            region=region,
            split="fixture",
            artifact_group_id=gid,
            source="comprehensive_fixture_builder",
        ))

    return samples


def _draw_artifact(path: Path, label: str, color: tuple[int, int, int]) -> None:
    """Draw a simple synthetic artifact image."""
    img = Image.new("RGB", (320, 320), color=(200, 195, 185))
    draw = ImageDraw.Draw(img)
    r, g, b = color

    # Draw a representative shape
    cx, cy = 160, 160
    # Body rectangle/ellipse
    draw.rectangle((cx - 50, cy - 60, cx + 50, cy + 80), outline=(r, g, b), width=3)
    # Top/neck
    draw.rectangle((cx - 20, cy - 100, cx + 20, cy - 60), outline=(r, g, b), width=2)
    # Decorative band
    draw.line((cx - 48, cy - 20, cx + 48, cy - 20), fill=(r, g, b), width=2)
    draw.line((cx - 48, cy + 20, cx + 48, cy + 20), fill=(r, g, b), width=2)
    # Label text
    draw.text((cx - 40, cy), label, fill=(r - 30, g - 20, b - 20))
    draw.text((cx - 30, cy + 20), f"({color})", fill=(150, 150, 150))

    img.save(path, "PNG")
