from __future__ import annotations

import pytest
import yaml

from histodelib.experiments.matrix import plan_experiments


def test_experiment_matrix_expands_methods_and_configs(tmp_path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "fixture-check",
                "methods": ["direct_vlm", "histodelib_rule"],
                "configs": ["configs/run/fixture.yaml"],
                "formal_dataset": "NOT_SELECTED",
            }
        ),
        encoding="utf-8",
    )

    plans = plan_experiments(matrix)

    assert [plan.method for plan in plans] == ["direct_vlm", "histodelib_rule"]
    assert all(plan.formal_dataset == "NOT_SELECTED" for plan in plans)


def test_experiment_matrix_rejects_formal_execution_without_authorization(tmp_path) -> None:
    matrix = tmp_path / "formal.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "formal",
                "methods": ["direct_vlm"],
                "configs": ["configs/run/fixture.yaml"],
                "formal_dataset": "some-dataset",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError):
        plan_experiments(matrix)
