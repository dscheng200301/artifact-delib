from __future__ import annotations

from pathlib import Path

from histodelib.data.fixture_builder import build_fixture
from histodelib.data.importer import import_manifest
from histodelib.data.leakage import find_group_leakage
from histodelib.data.splitter import split_samples
from histodelib.data.validator import validate_samples
from histodelib.schemas import Label, Sample


def test_fixture_builder_creates_labelled_synthetic_images(tmp_path: Path) -> None:
    samples = build_fixture(tmp_path)

    assert {sample.label for sample in samples} == set(Label)
    assert len(samples) >= 10
    assert all(sample.image_path.exists() for sample in samples)
    assert all(sample.is_synthetic_fixture for sample in samples)
    assert validate_samples(samples).is_valid is True


def test_sample_supports_fixture_and_unassigned_splits(tmp_path: Path) -> None:
    sample = Sample(
        sample_id="unassigned",
        image_path=tmp_path / "sample.png",
        caption="caption",
        split="fixture",
    )
    assert sample.split == "fixture"
    assert sample.model_copy(update={"split": "unassigned"}).split == "unassigned"


def test_validator_rejects_duplicate_ids_and_missing_images(tmp_path: Path) -> None:
    missing = Sample(
        sample_id="duplicate",
        image_path=tmp_path / "missing.png",
        caption="caption",
        label=Label.TRUE,
        fixture_markers={"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"},
    )
    duplicate = missing.model_copy(update={"image_path": tmp_path / "also-missing.png"})

    report = validate_samples([missing, duplicate])

    assert report.is_valid is False
    assert "duplicate sample_id: duplicate" in report.errors
    assert any("image does not exist" in error for error in report.errors)


def test_group_leakage_flags_group_in_multiple_splits(tmp_path: Path) -> None:
    first = Sample(
        sample_id="one",
        image_path=tmp_path / "one.png",
        caption="one",
        label=Label.TRUE,
        original_group_id="same-original",
        split="train",
    )
    second = first.model_copy(update={"sample_id": "two", "split": "test"})

    assert find_group_leakage([first, second]) == {"same-original": {"train", "test"}}


def test_manifest_import_and_group_aware_split(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,caption,label,original_group_id\n"
        f"{sample.sample_id},{sample.image_path.name},caption,TRUE,group-a\n",
        encoding="utf-8",
    )

    imported = import_manifest(manifest, tmp_path / "images")
    split = split_samples(imported, seed=7)

    assert imported[0].image_path == sample.image_path
    assert set(split) == {"train", "validation", "test"}
    assert sum(len(items) for items in split.values()) == 1


def test_manifest_import_rejects_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,caption,label\nunsafe,../secret.png,caption,TRUE\n",
        encoding="utf-8",
    )

    try:
        import_manifest(manifest, tmp_path / "images")
    except ValueError as exc:
        assert "outside image root" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")
