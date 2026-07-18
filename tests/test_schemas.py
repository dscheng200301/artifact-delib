from __future__ import annotations

import pytest
from pydantic import ValidationError

from histodelib.schemas import (
    ClaimFactPair,
    ImageEvidence,
    Label,
    PromptProvenance,
    Sample,
    TextEvidence,
    TokenUsage,
)


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


def test_modal_evidence_requires_provenance_and_keeps_uncertainty_explicit() -> None:
    provenance = PromptProvenance(name="text_agent", version="v1", content_hash="abc")
    text = TextEvidence(
        evidence_id="text-1",
        caption_claims=("A harbor in 1912",),
        uncertainty=("date is inferred",),
        prompt=provenance,
    )
    image = ImageEvidence(
        evidence_id="image-1",
        visible_scene=("harbor",),
        visible_text=None,
        uncertainty=("date not visible",),
        prompt=PromptProvenance(name="image_agent", version="v1", content_hash="def"),
    )

    pair = ClaimFactPair(
        claim="A harbor in 1912",
        visual_fact="harbor",
        relation="uncertain",
        conflict_type=None,
        evidence_ids=(text.evidence_id, image.evidence_id),
    )

    assert pair.relation == "uncertain"
    assert image.visible_text is None
