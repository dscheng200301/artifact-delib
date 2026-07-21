"""DINOv2 k-NN Baseline — external baseline.

Train Images → DINOv2 Features → Feature Index
Test Image → DINOv2 Feature → k Nearest Neighbors → Distance-Weighted Voting → Prediction

Requires optional dependencies: torch, transformers, timm, scikit-learn.

NOTE: This is a STUB implementation. Full functionality requires
optional vision dependencies.
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
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        raise ImportError(
            "DINOv2 k-NN baseline requires: " + ", ".join(missing) + ". "
            "Install with: pip install torch torchvision scikit-learn"
        )


class Dinov2KNNBaseline:
    """DINOv2 k-NN classification with distance-weighted voting.

    Builds a feature index from training images only. Never uses test images
    for index construction.

    NOTE: STUB — full implementation pending optional dependencies.
    """

    name = "dinov2_knn"

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        k: int = 5,
    ) -> None:
        _check_deps()
        self.model_name = model_name
        self.k = k
        self._model = None
        self._features: list = []
        self._labels: list[str] = []
        self._index_built = False

    def build_index(
        self,
        image_paths: list[Path],
        labels: list[str] | None = None,
    ) -> None:
        """Build feature index from training images."""
        raise NotImplementedError(
            "DINOv2 k-NN is not fully implemented in this phase. "
            "See STUB for the interface."
        )

    def is_index_built(self) -> bool:
        return self._index_built

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        raise NotImplementedError(
            "DINOv2 k-NN is not fully implemented in this phase."
        )
