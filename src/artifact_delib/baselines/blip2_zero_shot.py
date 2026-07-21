"""BLIP-2 Zero-Shot Baseline — external baseline.

Image → BLIP-2 → Natural-Language Identification → Parser → Structured Prediction

Requires optional dependencies: torch, transformers.

NOTE: This is a STUB implementation. Full functionality requires
optional vision-language dependencies.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.schemas import PipelineResult


def _check_deps() -> None:
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        raise ImportError(
            "BLIP-2 Zero-Shot baseline requires: " + ", ".join(missing) + ". "
            "Install with: pip install torch transformers Pillow"
        )


class Blip2ZeroShotBaseline:
    """BLIP-2 image-to-text identification baseline.

    Uses a single BLIP-2 call to produce an NL identification, then parses
    it with the shared PredictionParser. No expert decomposition, no routing.

    NOTE: STUB — full implementation pending optional dependencies.
    """

    name = "blip2_zero_shot"

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
    ) -> None:
        _check_deps()
        self.model_name = model_name

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        raise NotImplementedError(
            "BLIP-2 Zero-Shot is not fully implemented in this phase."
        )
