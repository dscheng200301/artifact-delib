from __future__ import annotations

from histodelib.evaluation.statistics import group_bootstrap_mean, paired_accuracy_delta


def test_group_bootstrap_is_deterministic_and_group_aware() -> None:
    first = group_bootstrap_mean([0.0, 1.0, 1.0], ["g1", "g1", "g2"], n_resamples=20)
    second = group_bootstrap_mean([0.0, 1.0, 1.0], ["g1", "g1", "g2"], n_resamples=20)

    assert first == second
    assert first[0] == 2 / 3


def test_paired_accuracy_delta_uses_shared_ids() -> None:
    assert paired_accuracy_delta({"a": True, "b": False}, {"a": False, "c": True}) == 1.0
