from __future__ import annotations

import pytest

from histodelib.config_schema import validate_runtime_config


def test_runtime_config_coerces_and_preserves_known_runtime_values() -> None:
    config = validate_runtime_config(
        {
            "name": "smoke",
            "mode": "api",
            "router": "api",
            "max_cross_exam_rounds": 1,
            "max_reinspection_targets": 2,
            "provenance_note": "kept",
        }
    )

    assert config["router"] == "api"
    assert config["max_cross_exam_rounds"] == 1
    assert config["provenance_note"] == "kept"


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "remote"},
        {"router": "unknown"},
        {"max_cross_exam_rounds": -1},
        {"max_reinspection_targets": -1},
    ],
)
def test_runtime_config_rejects_unsafe_values(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_runtime_config(values)
