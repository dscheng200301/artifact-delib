"""Tests for ArtifactDelib data layer — import, split, validate, fixtures, batch runner."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from artifact_delib.data.batch_runner import BatchRunner
from artifact_delib.data.fixture_builder import build_comprehensive_fixtures
from artifact_delib.data.importer import ArtifactDatasetImporter
from artifact_delib.data.splitter import ArtifactDatasetSplitter
from artifact_delib.data.validator import ArtifactDatasetValidator
from artifact_delib.models.mock_artifact import ArtifactMockClient
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import ArtifactSample


# ═══════════════════════════════════════════════════════════════
#  ArtifactDatasetImporter tests
# ═══════════════════════════════════════════════════════════════

def test_importer_reads_valid_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        # Create a dummy image
        img = t / "img-001.jpg"
        img.write_bytes(b"fake")

        # Create manifest
        manifest = t / "manifest.csv"
        manifest.write_text(
            "sample_id,image_path,category,fine_grained_type,period,artifact_group_id\n"
            "s1,img-001.jpg,瓷器,梅瓶,明永乐,g-001\n",
            encoding="utf-8",
        )

        importer = ArtifactDatasetImporter(image_root=t)
        samples = importer.import_manifest(manifest)

        assert len(samples) == 1
        assert samples[0].sample_id == "s1"
        assert samples[0].category == "瓷器"
        assert samples[0].fine_grained_type == "梅瓶"
        assert samples[0].artifact_group_id == "g-001"


def test_importer_rejects_duplicate_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        img = t / "img.jpg"
        img.write_bytes(b"fake")

        manifest = t / "manifest.csv"
        manifest.write_text(
            "sample_id,image_path\ns1,img.jpg\ns1,img.jpg\n", encoding="utf-8",
        )

        importer = ArtifactDatasetImporter(image_root=t)
        with pytest.raises(ValueError, match="duplicate"):
            importer.import_manifest(manifest)


def test_importer_detects_missing_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        manifest = t / "manifest.csv"
        manifest.write_text("sample_id,extra\ns1,val\n", encoding="utf-8")

        importer = ArtifactDatasetImporter(image_root=t)
        with pytest.raises(ValueError, match="missing"):
            importer.import_manifest(manifest)


# ═══════════════════════════════════════════════════════════════
#  ArtifactDatasetSplitter tests
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_list() -> list[ArtifactSample]:
    """Create a multi-group sample list for split testing."""
    samples = []
    for gid in range(10):
        for view in range(3):
            samples.append(ArtifactSample(
                sample_id=f"s-{gid}-{view}",
                image_path=Path(f"/tmp/img-{gid}-{view}.jpg"),
                category="瓷器",
                fine_grained_type="梅瓶",
                period="明",
                artifact_group_id=f"g-{gid:03d}",
            ))
    return samples


def test_splitter_preserves_groups(sample_list) -> None:
    """All views of the same artifact must be in the same split."""
    splitter = ArtifactDatasetSplitter(train_ratio=0.7, validation_ratio=0.1, seed=42)
    result = splitter.split(sample_list)

    all_samples = result["train"] + result["validation"] + result["test"]
    for gid in [f"g-{i:03d}" for i in range(10)]:
        group_samples = [s for s in all_samples if s.artifact_group_id == gid]
        splits = set(s.split for s in group_samples)
        assert len(splits) == 1, f"group {gid} in multiple splits: {splits}"


def test_splitter_respects_ratios(sample_list) -> None:
    """Split should approximately match requested ratios."""
    splitter = ArtifactDatasetSplitter(train_ratio=0.7, validation_ratio=0.1, seed=42)
    result = splitter.split(sample_list)

    n_total = len(sample_list)
    assert ratio_approx(len(result["train"]) / n_total, 0.7, epsilon=0.15)
    assert ratio_approx(len(result["validation"]) / n_total, 0.1, epsilon=0.1)


def test_splitter_summary(sample_list) -> None:
    splitter = ArtifactDatasetSplitter()
    result = splitter.split(sample_list)
    summary = splitter.summary(result)
    assert "train" in summary
    assert "unique objects" in summary


def ratio_approx(a: float, b: float, epsilon: float = 0.2) -> bool:
    return abs(a - b) <= epsilon


# ═══════════════════════════════════════════════════════════════
#  ArtifactDatasetValidator tests
# ═══════════════════════════════════════════════════════════════

def test_validator_detects_missing_images() -> None:
    validator = ArtifactDatasetValidator()
    samples = [ArtifactSample(
        sample_id="s1", image_path=Path("/nonexistent/img.jpg"),
    )]
    issues = validator.validate(samples)
    assert len(issues["missing_images"]) == 1


def test_validator_detects_duplicates() -> None:
    validator = ArtifactDatasetValidator()
    samples = [
        ArtifactSample(sample_id="s1", image_path=Path("/tmp/a.jpg")),
        ArtifactSample(sample_id="s1", image_path=Path("/tmp/b.jpg")),
    ]
    issues = validator.validate(samples)
    assert len(issues["duplicate_ids"]) == 1


def test_validator_detects_group_leakage() -> None:
    validator = ArtifactDatasetValidator()
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
    issues = validator.validate(samples)
    assert len(issues["group_split_leakage"]) == 1


def test_validator_report_runs(sample_list) -> None:
    validator = ArtifactDatasetValidator()
    report = validator.report(sample_list)
    assert "Total samples" in report


# ═══════════════════════════════════════════════════════════════
#  Comprehensive fixture builder tests
# ═══════════════════════════════════════════════════════════════

def test_comprehensive_fixtures_24_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        samples = build_comprehensive_fixtures(Path(tmp))
        assert len(samples) >= 20
        assert all(s.image_path.exists() for s in samples)
        assert all(s.image_path.stat().st_size > 0 for s in samples)


def test_comprehensive_fixtures_categories() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        samples = build_comprehensive_fixtures(Path(tmp))
        cats = set(s.category for s in samples)
        expected = {"瓷器", "青铜器", "玉器", "漆器"}
        assert expected.issubset(cats), f"Missing categories: {expected - cats}"


def test_comprehensive_fixtures_groups() -> None:
    """Multiple views per group should exist for some artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        samples = build_comprehensive_fixtures(Path(tmp))
        from collections import Counter
        gid_counts = Counter(s.artifact_group_id for s in samples)
        # At least one group should have multiple views
        assert any(c > 1 for c in gid_counts.values()), "No multi-view groups found"


# ═══════════════════════════════════════════════════════════════
#  BatchRunner tests
# ═══════════════════════════════════════════════════════════════

def test_batch_runner_runs_on_fixtures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        samples = build_comprehensive_fixtures(t)

        pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
        runner = BatchRunner(
            method=pipeline,
            output_root=t / "output",
            experiment_id="test-batch-001",
            method_name="artifact_delib_rule",
        )

        results = runner.run(samples[:4])  # Run first 4
        assert len(results) == 4
        assert all(r.status == "COMPLETED" for r in results)


def test_batch_runner_evaluates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        samples = build_comprehensive_fixtures(t)

        pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())
        runner = BatchRunner(
            method=pipeline,
            output_root=t / "output",
            experiment_id="test-batch-002",
            method_name="artifact_delib_rule",
        )

        results = runner.run(samples[:4])
        metrics = runner.evaluate(results, samples[:4])

        assert "top1_type_accuracy" in metrics
        assert "top5_type_accuracy" in metrics
        assert metrics["n_samples"] == 4


def test_batch_runner_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        samples = build_comprehensive_fixtures(t)

        pipeline = ArtifactDelibPipeline(client=ArtifactMockClient())

        # First run
        runner1 = BatchRunner(
            method=pipeline, output_root=t / "output",
            experiment_id="test-resume-001", method_name="test",
        )
        r1 = runner1.run(samples[:2])

        # Second run (should skip completed)
        runner2 = BatchRunner(
            method=pipeline, output_root=t / "output",
            experiment_id="test-resume-001", method_name="test",
        )
        r2 = runner2.run(samples[:4])

        # All 4 should complete (first 2 from cache)
        assert len(r2) == 2  # Only uncompleted ones returned


def test_batch_runner_with_baselines() -> None:
    from artifact_delib.baselines import DirectVLMBaseline, FixedMultiExpertBaseline

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        samples = build_comprehensive_fixtures(t)

        for name, bl in [
            ("B1", DirectVLMBaseline(ArtifactMockClient())),
            ("B2", FixedMultiExpertBaseline(ArtifactMockClient())),
        ]:
            runner = BatchRunner(
                method=bl, output_root=t / f"output-{name}",
                experiment_id=f"test-{name}", method_name=name,
            )
            results = runner.run(samples[:2])
            assert all(r.status == "COMPLETED" for r in results)
