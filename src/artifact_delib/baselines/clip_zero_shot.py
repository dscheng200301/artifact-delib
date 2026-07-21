"""CLIP Zero-Shot Baseline — external baseline.

Image → CLIP Image Encoder → Label Prompt Embeddings → Cosine Similarity → Prediction

Uses frozen CLIP with a frozen label ontology. Multiple text templates per label
are averaged for robust zero-shot classification.

Supports prediction of: category, artifact type, dynasty/period, material.

Requires optional dependencies: torch, transformers, Pillow.
Lazy import — the module can be imported without these deps.
Model weights are NOT downloaded automatically unless allow_model_download=True.
"""

from __future__ import annotations

import hashlib
import json
import logging
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

# ── Frozen label ontology (NOT derived from test samples) ──

CATEGORY_LABELS = [
    "瓷器", "青铜器", "玉器", "漆器", "金银器",
    "陶器", "雕塑", "纺织品", "珐琅器", "书法", "绘画",
]

ARTIFACT_TYPE_LABELS = [
    "梅瓶", "玉壶春瓶", "天球瓶", "胆瓶", "瓶", "壶",
    "鼎", "簋", "爵", "尊", "彝", "觚", "斝", "卣",
    "盘", "碗", "杯", "盏", "碟", "钵",
    "罐", "缸", "瓮",
    "玉璧", "玉琮", "玉璋", "玉圭", "玉璜", "玉玦",
    "俑", "佛像", "菩萨像", "天王像",
    "镜", "炉", "灯", "洗", "砚", "枕", "盒",
    "如意", "香炉", "笔筒", "笔洗", "水盂",
    "屏风", "挂轴", "扇面", "册页",
]

PERIOD_LABELS = [
    "商代", "西周", "春秋", "战国", "秦",
    "西汉", "东汉", "三国", "晋", "南北朝",
    "隋", "唐", "五代", "宋", "辽", "金", "元", "明", "清",
]

MATERIAL_LABELS = [
    "瓷", "青铜", "玉", "漆", "金银", "陶", "珐琅",
    "石", "木", "丝", "牙骨", "玻璃", "铜鎏金",
]

# ── Prompt templates (one ensemble per label) ──

PROMPT_TEMPLATES = [
    "一张{label}的文物图片",
    "一件{label}",
    "古代{label}文物",
    "博物馆中的{label}",
    "这是{label}",
]


class ClipZeroShotBaseline:
    """CLIP zero-shot artifact identification.

    Uses frozen CLIP model. Downloads only when allow_model_download=True.
    Image features are cached to disk for repeated runs.

    Attributes:
        name: Canonical baseline name.
        model_name: HuggingFace model id (e.g. "openai/clip-vit-base-patch32").
        model_revision: Specific model revision (for reproducibility).
    """

    name = "clip_zero_shot"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_revision: str = "main",
        cache_dir: Path | None = None,
        allow_model_download: bool = False,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_dir = cache_dir or Path("data/clip_cache")
        self.allow_model_download = allow_model_download
        self.device = device

        self._model = None
        self._processor = None
        self._text_features: dict[str, dict[str, list[float]]] = {}  # label_set → label → embedding

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run CLIP zero-shot on a single image.

        Returns PipelineResult with parsed fields in final_identification.
        """
        import time

        t0 = time.perf_counter()
        usage = TokenUsage(input_tokens=0, output_tokens=0)

        # Ensure model loaded
        if self._model is None:
            self._load_model()

        # Get image embedding
        image_feature = self._encode_image(image_path)

        # Predict each dimension
        category = self._predict(image_feature, CATEGORY_LABELS, "category")
        artifact_type = self._predict(image_feature, ARTIFACT_TYPE_LABELS, "artifact_type")
        period = self._predict(image_feature, PERIOD_LABELS, "period")
        material = self._predict(image_feature, MATERIAL_LABELS, "material")

        latency_ms = (time.perf_counter() - t0) * 1000

        # Build NL identification
        parts = []
        if category:
            parts.append(f"文物大类：{category}")
        if artifact_type:
            parts.append(f"具体类型：{artifact_type}")
        if period:
            parts.append(f"年代时期：{period}")
        if material:
            parts.append(f"材质：{material}")

        content = "；".join(parts) if parts else "CLIP未能识别该文物。"
        final = FinalIdentification(content=content, usage=usage)

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(
                content=json.dumps({
                    "model": self.model_name,
                    "revision": self.model_revision,
                    "predicted_category": category,
                    "predicted_type": artifact_type,
                    "predicted_period": period,
                    "predicted_material": material,
                }, ensure_ascii=False),
                usage=TokenUsage(),
            ),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=usage,
            total_api_calls=0,
            status="COMPLETED",
        )

    # ── Internal helpers ──

    def _load_model(self) -> None:
        """Lazy-load CLIP model. Fail closed if download not allowed."""
        if not self.allow_model_download:
            # Check if model is cached locally
            cache_marker = self.cache_dir / "clip_model_loaded.txt"
            if not cache_marker.exists():
                raise RuntimeError(
                    "CLIP model download not allowed. "
                    "Set allow_model_download=True or pre-download the model."
                )

        try:
            import torch
        except ImportError:
            raise ImportError(
                "CLIP baseline requires PyTorch. Install: pip install torch torchvision"
            )

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "CLIP baseline requires transformers. Install: pip install transformers"
            )

        self._processor = CLIPProcessor.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model = CLIPModel.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model.eval()

        # Move to device
        if self.device != "cpu" and torch.cuda.is_available():
            self._model = self._model.to(self.device)

        # Ensure all text embeddings are precomputed
        for label_set_name, labels in [
            ("category", CATEGORY_LABELS),
            ("artifact_type", ARTIFACT_TYPE_LABELS),
            ("period", PERIOD_LABELS),
            ("material", MATERIAL_LABELS),
        ]:
            self._text_features[label_set_name] = self._compute_text_embeddings(labels)

        # Mark as loaded
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "clip_model_loaded.txt").touch()

    def _compute_text_embeddings(self, labels: list[str]) -> dict[str, list[float]]:
        """Compute averaged text embeddings for each label using all templates."""
        import torch

        embeddings: dict[str, list[float]] = {}
        for label in labels:
            prompts = [tpl.format(label=label) for tpl in PROMPT_TEMPLATES]
            inputs = self._processor(
                text=prompts, return_tensors="pt", padding=True,
            )
            if self.device != "cpu" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Average across templates
                avg_feature = text_features.mean(dim=0)
                avg_feature = avg_feature / avg_feature.norm()

            embeddings[label] = avg_feature.cpu().tolist()

        return embeddings

    def _encode_image(self, image_path: Path) -> list[float]:
        """Encode image into normalized CLIP feature vector."""
        import torch
        from PIL import Image

        cache_key = self._image_cache_key(image_path)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            return json.loads(cache_path.read_text())

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        if self.device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        feature = image_features[0].cpu().tolist()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(feature))
        return feature

    def _predict(
        self,
        image_feature: list[float],
        labels: list[str],
        label_set_name: str,
    ) -> str | None:
        """Predict the most similar label using cosine similarity."""
        import torch

        if label_set_name not in self._text_features:
            return None

        img_tensor = torch.tensor(image_feature)
        img_tensor = img_tensor / img_tensor.norm()

        best_label = None
        best_sim = -1.0
        for label, text_feat in self._text_features[label_set_name].items():
            text_tensor = torch.tensor(text_feat)
            text_tensor = text_tensor / text_tensor.norm()
            sim = torch.dot(img_tensor, text_tensor).item()
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label

    @staticmethod
    def _image_cache_key(image_path: Path) -> str:
        """Generate cache key from file path + size + mtime."""
        stat = image_path.stat()
        seed = f"{image_path.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
