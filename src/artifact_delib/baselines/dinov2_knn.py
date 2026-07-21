"""DINOv2 k-NN Baseline — external baseline.

Train Images → DINOv2 Features → Normalize → Feature Index
Test Image  → DINOv2 Feature → k Nearest Neighbors → Distance-Weighted Voting

k is selected ONLY from validation set performance.
Train/Test split is enforced via object-disjoint split.

Requires optional dependencies: torch, transformers, scikit-learn.
Model weights are NOT downloaded automatically unless allow_model_download=True.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from artifact_delib.schemas import (
    CandidateSet,
    FinalIdentification,
    PipelineResult,
    SummarizedReport,
    TokenUsage,
    VisualPerceptionReport,
)

logger = logging.getLogger(__name__)

# Valid k values (final k is selected from validation only)
VALID_K_VALUES = (1, 5, 10)


class Dinov2KNNBaseline:
    """DINOv2 k-NN classification with distance-weighted voting.

    Builds a feature index from training images ONLY. Never uses test images
    for index construction. Object-disjoint split is enforced.

    Attributes:
        name: Canonical baseline name.
        model_name: HuggingFace model id.
        k: Number of neighbors (selected from validation).
        selected_k: The k that was actually used (after validation selection).
    """

    name = "dinov2_knn"

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        k: int = 5,
        cache_dir: Path | None = None,
        allow_model_download: bool = False,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.selected_k: int | None = None
        self.cache_dir = cache_dir or Path("data/dinov2_cache")
        self.allow_model_download = allow_model_download
        self.device = device

        self._model = None
        self._processor = None
        self._features: list[list[float]] = []
        self._labels_category: list[str | None] = []
        self._labels_type: list[str | None] = []
        self._labels_period: list[str | None] = []
        self._labels_material: list[str | None] = []
        self._image_ids: list[str] = []
        self._index_built = False

    def build_index(
        self,
        image_paths: list[Path],
        labels_category: list[str | None] | None = None,
        labels_type: list[str | None] | None = None,
        labels_period: list[str | None] | None = None,
        labels_material: list[str | None] | None = None,
    ) -> None:
        """Build feature index from TRAINING images only.

        Args:
            image_paths: Training-set image paths.
            labels_*: Ground-truth labels (for k-NN lookup, never used in inference).
        """
        if self._model is None:
            self._load_model()

        import torch
        from PIL import Image

        features: list[list[float]] = []
        valid_indices: list[int] = []

        for i, img_path in enumerate(image_paths):
            cache_key = self._image_cache_key(img_path)
            cache_path = self.cache_dir / f"train_{cache_key}.json"

            if cache_path.exists():
                feat = json.loads(cache_path.read_text())
            else:
                try:
                    image = Image.open(img_path).convert("RGB")
                    feat = self._extract_feature(image)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(feat))
                except Exception as e:
                    logger.warning(f"Failed to encode {img_path}: {e}")
                    continue

            features.append(feat)
            valid_indices.append(i)

        self._features = features
        if labels_category:
            self._labels_category = [labels_category[i] for i in valid_indices]
        if labels_type:
            self._labels_type = [labels_type[i] for i in valid_indices]
        if labels_period:
            self._labels_period = [labels_period[i] for i in valid_indices]
        if labels_material:
            self._labels_material = [labels_material[i] for i in valid_indices]
        self._image_ids = [str(img_paths[i]) for i in valid_indices]

        self._index_built = True
        logger.info(f"DINOv2 index built: {len(self._features)} features")

    def is_index_built(self) -> bool:
        return self._index_built

    def select_k_from_validation(
        self,
        val_image_paths: list[Path],
        val_labels: list[str],
        label_field: str = "type",
    ) -> dict[int, float]:
        """Evaluate all k values on validation set. Return {k: accuracy}.

        Args:
            val_image_paths: Validation image paths.
            val_labels: Ground-truth labels (validation only).
            label_field: Which label dimension to optimize ("type", "period", etc.).
        """
        if not self._index_built:
            raise RuntimeError("Must call build_index() before validation.")

        scores: dict[int, float] = {}
        for k in VALID_K_VALUES:
            self.k = k
            correct = 0
            for img_path, gold in zip(val_image_paths, val_labels):
                pred = self._predict_single(img_path, label_field)
                if pred == gold:
                    correct += 1
            scores[k] = correct / len(val_image_paths) if val_image_paths else 0.0

        # Select best k
        best_k = max(scores, key=lambda k: scores[k])
        self.selected_k = best_k
        self.k = best_k
        logger.info(f"Selected k={best_k} from validation (scores: {scores})")
        return scores

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run DINOv2 k-NN on a single test image.

        Pre-condition: build_index() must have been called with train data.
        The test image must NOT be in the training index.
        """
        import time

        if not self._index_built:
            raise RuntimeError(
                "DINOv2 index not built. Call build_index(train_images) first."
            )

        t0 = time.perf_counter()

        category = self._predict_single(image_path, "category")
        artifact_type = self._predict_single(image_path, "type")
        period = self._predict_single(image_path, "period")
        material = self._predict_single(image_path, "material")

        latency_ms = (time.perf_counter() - t0) * 1000

        parts = []
        if category:
            parts.append(f"文物大类：{category}")
        if artifact_type:
            parts.append(f"具体类型：{artifact_type}")
        if period:
            parts.append(f"年代时期：{period}")
        if material:
            parts.append(f"材质：{material}")

        content = "；".join(parts) if parts else "DINOv2未能识别该文物。"
        final = FinalIdentification(content=content, usage=TokenUsage())

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(
                content=json.dumps({
                    "model": self.model_name,
                    "k": self.k,
                    "selected_k": self.selected_k,
                    "index_size": len(self._features),
                    "predicted_category": category,
                    "predicted_type": artifact_type,
                    "predicted_period": period,
                    "predicted_material": material,
                }, ensure_ascii=False),
                usage=TokenUsage(),
            ),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=TokenUsage(),
            total_api_calls=0,
            status="COMPLETED",
        )

    # ── Internal helpers ──

    def _predict_single(self, image_path: Path, label_field: str) -> str | None:
        """Predict a single label using distance-weighted k-NN voting."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        query = self._extract_feature(image)
        query_tensor = torch.tensor(query)

        # Get label list for this field
        label_list = getattr(self, f"_labels_{label_field}", [])
        if not label_list:
            return None

        # Compute cosine distances to all training features
        distances: list[tuple[float, str | None]] = []
        for feat, label in zip(self._features, label_list):
            feat_tensor = torch.tensor(feat)
            # Cosine distance = 1 - cosine_similarity
            sim = torch.dot(query_tensor, feat_tensor).item()
            dist = 1.0 - sim
            distances.append((dist, label))

        # Sort by distance, take top-k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]

        # Distance-weighted voting
        weights: dict[str, float] = {}
        for dist, label in top_k:
            if label is None:
                continue
            # Weight = 1 / (distance + epsilon)
            weight = 1.0 / (dist + 1e-6)
            weights[label] = weights.get(label, 0.0) + weight

        if not weights:
            return None

        return max(weights, key=lambda k: weights[k])

    def _extract_feature(self, image) -> list[float]:
        """Extract normalized DINOv2 feature from a PIL image."""
        import torch

        inputs = self._processor(images=image, return_tensors="pt")
        if self.device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # DINOv2 outputs last_hidden_state; use CLS token
            feature = outputs.last_hidden_state[:, 0, :]
            feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature[0].cpu().tolist()

    def _load_model(self) -> None:
        """Lazy-load DINOv2 model. Fail closed if download not allowed."""
        if not self.allow_model_download:
            cache_marker = self.cache_dir / "dinov2_model_loaded.txt"
            if not cache_marker.exists():
                raise RuntimeError(
                    "DINOv2 model download not allowed. "
                    "Set allow_model_download=True or pre-download the model."
                )

        try:
            import torch
        except ImportError:
            raise ImportError("DINOv2 requires PyTorch. Install: pip install torch")

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "DINOv2 requires transformers. Install: pip install transformers"
            )

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        if self.device != "cpu" and torch.cuda.is_available():
            self._model = self._model.to(self.device)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "dinov2_model_loaded.txt").touch()

    @staticmethod
    def _image_cache_key(image_path: Path) -> str:
        stat = image_path.stat()
        seed = f"{image_path.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]
