"""Validation for manifest-derived samples without accessing external datasets."""

from __future__ import annotations

from PIL import Image
from pydantic import BaseModel, ConfigDict, computed_field

from histodelib.schemas import Sample


class ValidationReport(BaseModel):
    """Validation outcome with human-readable errors."""

    model_config = ConfigDict(frozen=True)

    errors: list[str]

    @computed_field(return_type=bool)  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_samples(samples: list[Sample]) -> ValidationReport:
    """Check IDs, image readability prerequisites, captions, and safe paths."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if not sample.image_path.exists():
            errors.append(f"image does not exist: {sample.image_path}")
        elif not sample.image_path.is_file():
            errors.append(f"image path is not a file: {sample.image_path}")
        else:
            try:
                with Image.open(sample.image_path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                errors.append(f"image is unreadable: {sample.sample_id}: {exc}")
        if not sample.caption.strip():
            errors.append(f"caption is blank: {sample.sample_id}")
    return ValidationReport(errors=errors)
