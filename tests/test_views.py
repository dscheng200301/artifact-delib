from __future__ import annotations

from pathlib import Path

import pytest

from histodelib.data.fixture_builder import build_fixture
from histodelib.vision.views import create_view


def test_create_view_is_deterministic_and_records_hashes(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    first = create_view(sample.image_path, "patch", bbox=(0, 0, 8, 8))
    second = create_view(sample.image_path, "patch", bbox=(0, 0, 8, 8))

    assert first.source_sha256 == second.source_sha256
    assert first.view_sha256 == second.view_sha256
    assert first.bbox == (0, 0, 8, 8)
    assert first.data == second.data


def test_create_view_rejects_out_of_bounds_bbox(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    with pytest.raises(ValueError, match="bbox"):
        create_view(sample.image_path, "patch", bbox=(-1, 0, 8, 8))
