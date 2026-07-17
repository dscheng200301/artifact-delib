from __future__ import annotations

import pytest
from pydantic import ValidationError

from histodelib.schemas import Label, Sample, TokenUsage


def test_sample_accepts_required_fixture_markers() -> None:
    sample = Sample(
        sample_id="fixture-001",
        image_path="fixtures/images/fixture-001.png",
        caption="A synthetic caption.",
        label=Label.TRUE,
        fixture_markers={"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"},
    )

    assert sample.label is Label.TRUE
    assert sample.is_synthetic_fixture is True


def test_sample_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        Sample(
            sample_id="fixture-001",
            image_path="fixtures/images/fixture-001.png",
            caption="A synthetic caption.",
            label="MAYBE",
            fixture_markers={"SYNTHETIC_FIXTURE", "NOT_FOR_RESEARCH_RESULTS"},
        )


def test_token_usage_adds_input_and_output_tokens() -> None:
    usage = TokenUsage(input_tokens=13, output_tokens=5)

    assert usage.total_tokens == 18
