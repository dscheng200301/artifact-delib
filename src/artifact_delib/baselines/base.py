"""Protocol definitions for baseline runners.

All external baselines and ablation variants implement this protocol
so they can be used interchangeably with BatchRunner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from artifact_delib.schemas import PipelineResult


@runtime_checkable
class BaselineProtocol(Protocol):
    """Unified protocol for all pipeline/baseline/ablation runners.

    Every method must expose:
      - name: human-readable identifier
      - run(image_path, sample_id) -> PipelineResult

    Methods that need a training/feature-index step expose:
      - build_index(image_paths, labels) -> None
    """

    name: str

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run this method on a single artifact image.

        Args:
            image_path: Path to the artifact image file.
            sample_id: Unique identifier for the sample (used in result).

        Returns:
            PipelineResult with final identification and provenance.
        """
        ...


class IndexableBaseline(BaselineProtocol):
    """Extension for baselines that need a training-set index (k-NN, etc.)."""

    def build_index(
        self,
        image_paths: list[Path],
        labels: list[str] | None = None,
    ) -> None:
        """Build a search index from training images.

        Args:
            image_paths: Paths to training-set images.
            labels: Optional ground-truth labels (only used for index, never leaked).
        """
        ...

    def is_index_built(self) -> bool:
        """Return True if the index has been built and is ready for inference."""
        ...
