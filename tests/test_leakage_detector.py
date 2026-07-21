"""Tests for LeakageDetector — near-duplicate, object overlap, and preflight integration.

These tests complement the existing leakage tests in test_artifact_delib_comprehensive.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from artifact_delib.data.leakage_detector import LeakageDetector
from artifact_delib.experiment.preflight import _detect_object_overlap, run_leakage_preflight
from artifact_delib.schemas import ArtifactSample


# ═══════════════════════════════════════════════════════════════
#  Near-duplicate detection (perceptual hash)
# ═══════════════════════════════════════════════════════════════


def test_near_duplicate_detects_same_image() -> None:
    """The same image file should be a near-duplicate with distance 0."""
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img = tmp_path / "test.png"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img)

        samples = [
            ArtifactSample(sample_id="s1", image_path=img),
            ArtifactSample(sample_id="s2", image_path=img),
        ]
        result = LeakageDetector.detect_near_duplicates(samples, threshold=5)
        assert len(result) >= 1
        # Distance should be 0 for identical images
        assert result[0].hamming_distance == 0


def test_near_duplicate_detects_tiny_modification() -> None:
    """A tiny pixel modification should produce a near-duplicate with small distance."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img1_path = tmp_path / "original.png"
        img2_path = tmp_path / "modified.png"

        # Create a simple image
        img1 = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img1.save(img1_path)

        # Create a slightly modified version (change 5 pixels)
        img2 = img1.copy()
        pixels = img2.load()
        assert pixels is not None
        pixels[0, 0] = (101, 151, 201)
        pixels[1, 1] = (102, 152, 202)
        pixels[2, 2] = (103, 153, 203)
        pixels[3, 3] = (104, 154, 204)
        pixels[4, 4] = (105, 155, 205)
        img2.save(img2_path)

        samples = [
            ArtifactSample(sample_id="s1", image_path=img1_path),
            ArtifactSample(sample_id="s2", image_path=img2_path),
        ]
        result = LeakageDetector.detect_near_duplicates(samples, threshold=5)
        assert len(result) >= 1, "Tiny pixel modification should be detected as near-duplicate"
        assert result[0].hamming_distance <= 5


def test_near_duplicate_different_images() -> None:
    """Completely different images should not be near-duplicates."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def make_horizontal_stripes() -> Image.Image:
            img = Image.new("RGB", (64, 64))
            pix = img.load()
            for y in range(64):
                val = 255 if y % 2 == 0 else 0
                for x in range(64):
                    pix[x, y] = (val, val, val)
            return img

        def make_vertical_stripes() -> Image.Image:
            img = Image.new("RGB", (64, 64))
            pix = img.load()
            for x in range(64):
                val = 255 if x % 2 == 0 else 0
                for y in range(64):
                    pix[x, y] = (val, val, val)
            return img

        img1_path = tmp_path / "horizontal.png"
        img2_path = tmp_path / "vertical.png"
        make_horizontal_stripes().save(img1_path)
        make_vertical_stripes().save(img2_path)

        samples = [
            ArtifactSample(sample_id="s1", image_path=img1_path),
            ArtifactSample(sample_id="s2", image_path=img2_path),
        ]
        result = LeakageDetector.detect_near_duplicates(samples, threshold=5)
        # Horizontal and vertical stripes should have a large Hamming distance
        assert len(result) == 0, "Different patterns should not be near-duplicates"


def test_near_duplicate_threshold_respected() -> None:
    """Threshold of 0 should only find identical images."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img1_path = tmp_path / "a.png"
        img2_path = tmp_path / "b.png"

        # Two completely different images
        def make_horizontal_stripes() -> Image.Image:
            img = Image.new("RGB", (64, 64))
            pix = img.load()
            for y in range(64):
                val = 255 if y % 2 == 0 else 0
                for x in range(64):
                    pix[x, y] = (val, val, val)
            return img

        def make_vertical_stripes() -> Image.Image:
            img = Image.new("RGB", (64, 64))
            pix = img.load()
            for x in range(64):
                val = 255 if x % 2 == 0 else 0
                for y in range(64):
                    pix[x, y] = (val, val, val)
            return img

        make_horizontal_stripes().save(img1_path)
        make_vertical_stripes().save(img2_path)

        samples = [
            ArtifactSample(sample_id="s1", image_path=img1_path),
            ArtifactSample(sample_id="s2", image_path=img2_path),
        ]
        # With threshold=0, only identical images match
        result = LeakageDetector.detect_near_duplicates(samples, threshold=0)
        assert len(result) == 0, "Threshold 0 should not find different patterns as near-duplicate"


# ═══════════════════════════════════════════════════════════════
#  Object / group overlap detection
# ═══════════════════════════════════════════════════════════════


def test_detect_object_overlap_finds_cross_split() -> None:
    """Same artifact_group_id in different splits should be detected."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            artifact_group_id="g1", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            artifact_group_id="g1", split="test",
        ),
    ]
    overlapping = _detect_object_overlap(samples)
    assert len(overlapping) == 2
    assert "s1" in overlapping
    assert "s2" in overlapping


def test_detect_object_overlap_no_overlap() -> None:
    """Different groups in same split should not be flagged."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            artifact_group_id="g1", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            artifact_group_id="g2", split="train",
        ),
    ]
    overlapping = _detect_object_overlap(samples)
    assert len(overlapping) == 0


def test_detect_object_overlap_no_group_id() -> None:
    """Samples without artifact_group_id should not raise errors."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"), split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"), split="test",
        ),
    ]
    overlapping = _detect_object_overlap(samples)
    assert len(overlapping) == 0


def test_detect_object_overlap_three_way() -> None:
    """Overlap across train/validation/test should all be flagged."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            artifact_group_id="g1", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            artifact_group_id="g1", split="validation",
        ),
        ArtifactSample(
            sample_id="s3", image_path=Path("/tmp/c.jpg"),
            artifact_group_id="g1", split="test",
        ),
    ]
    overlapping = _detect_object_overlap(samples)
    assert len(overlapping) == 3


# ═══════════════════════════════════════════════════════════════
#  Leakage preflight integration
# ═══════════════════════════════════════════════════════════════


def test_leakage_preflight_fails_on_object_overlap() -> None:
    """Leakage preflight should fail when fail_on_object_overlap is True."""
    from artifact_delib.experiment.config import LeakageConfig

    config = LeakageConfig(
        run_before_experiment=True,
        fail_on_exact_duplicate=True,
        fail_on_object_overlap=True,
    )
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            artifact_group_id="g1", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            artifact_group_id="g1", split="test",
        ),
    ]
    report = run_leakage_preflight(samples, config)
    assert not report.passed
    assert any("OBJECT_OVERLAP" in c for c in report.checks)


def test_leakage_preflight_warns_on_near_duplicate() -> None:
    """Leakage preflight should warn when near-duplicates found, if configured."""
    from PIL import Image
    from artifact_delib.experiment.config import LeakageConfig

    config = LeakageConfig(
        run_before_experiment=True,
        warn_on_near_duplicate=True,
        fail_on_exact_duplicate=False,  # Don't fail on exact duplicates
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Use two different images that are still somewhat similar
        def make_gradient(angle: str) -> Image.Image:
            img = Image.new("RGB", (64, 64))
            pix = img.load()
            for x in range(64):
                for y in range(64):
                    if angle == "horizontal":
                        val = int((x / 64.0) * 200) + 28
                    else:
                        val = int((y / 64.0) * 200) + 28
                    pix[x, y] = (val, val, val)
            return img

        img1_path = tmp_path / "a.png"
        img2_path = tmp_path / "b.png"
        make_gradient("horizontal").save(img1_path)
        make_gradient("vertical").save(img2_path)

        samples = [
            ArtifactSample(sample_id="s1", image_path=img1_path),
            ArtifactSample(sample_id="s2", image_path=img2_path),
        ]
        report = run_leakage_preflight(samples, config)
        # Should pass (warn_only, not fail)
        assert report.passed, f"Preflight should pass with warnings only, errors: {report.errors}"


# ═══════════════════════════════════════════════════════════════
#  Label coverage check
# ═══════════════════════════════════════════════════════════════


def test_label_coverage_missing_labels() -> None:
    """Test labels not in train should be reported."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            category="瓷器", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            category="玉器", split="test",
        ),
    ]
    result = LeakageDetector.check_label_coverage(samples)
    assert result is not None
    assert not result.is_clean
    assert "玉器" in result.missing_test_labels


def test_label_coverage_clean() -> None:
    """All test labels in train should pass."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"),
            category="瓷器", split="train",
        ),
        ArtifactSample(
            sample_id="s2", image_path=Path("/tmp/b.jpg"),
            category="瓷器", split="test",
        ),
    ]
    result = LeakageDetector.check_label_coverage(samples)
    assert result is not None
    assert result.is_clean
    assert len(result.missing_test_labels) == 0


def test_label_coverage_no_splits() -> None:
    """No split assignments should return None."""
    samples = [
        ArtifactSample(
            sample_id="s1", image_path=Path("/tmp/a.jpg"), category="瓷器",
        ),
    ]
    result = LeakageDetector.check_label_coverage(samples)
    assert result is None


# ═══════════════════════════════════════════════════════════════
#  run_all integration
# ═══════════════════════════════════════════════════════════════


def test_run_all_with_full_pipeline_scenario() -> None:
    """run_all should handle a realistic multi-sample scenario."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        img3 = tmp_path / "corrupt.png"

        # Use different patterns to avoid exact-duplicate or near-duplicate detection
        # Image with only top-left quad white
        def make_topleft_white() -> Image.Image:
            img = Image.new("RGB", (64, 64), color=(0, 0, 0))
            pix = img.load()
            for x in range(32):
                for y in range(32):
                    pix[x, y] = (255, 255, 255)
            return img

        # Image with only bottom-right quad white
        def make_bottomright_white() -> Image.Image:
            img = Image.new("RGB", (64, 64), color=(0, 0, 0))
            pix = img.load()
            for x in range(32, 64):
                for y in range(32, 64):
                    pix[x, y] = (255, 255, 255)
            return img

        make_topleft_white().save(img1)
        make_bottomright_white().save(img2)
        img3.write_bytes(b"not an image")

        samples = [
            ArtifactSample(
                sample_id="s1", image_path=img1, category="瓷器",
                split="train", artifact_group_id="g1",
            ),
            ArtifactSample(
                sample_id="s2", image_path=img2, category="玉器",
                split="test", artifact_group_id="g2",
            ),
            ArtifactSample(
                sample_id="s3", image_path=img3, category="青铜器",
                split="test",
            ),
        ]
        report = LeakageDetector.run_all(samples)
        assert report is not None
        # Should find the corrupt image
        assert len(report.corrupt_images) == 1
        assert report.corrupt_images[0].sample_id == "s3"
        # Summary should be a non-empty string
        assert len(report.summary) > 0