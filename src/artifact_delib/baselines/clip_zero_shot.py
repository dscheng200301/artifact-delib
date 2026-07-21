"""CLIP Zero-Shot Baseline — external baseline.

Image → CLIP Image Encoder → Label Prompt Embeddings → Cosine Similarity → Prediction

Requires optional dependencies: torch, transformers, Pillow.

This is lazily imported — the module can be imported without these dependencies,
but running the baseline will raise an ImportError if they are missing.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.schemas import PipelineResult


# ── Lazy import guard ──

def _check_deps() -> None:
    """Check that optional dependencies are available."""
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
            "CLIP Zero-Shot baseline requires: " + ", ".join(missing) + ". "
            "Install with: pip install torch transformers Pillow"
        )


# ── Label ontology (frozen, NOT derived from test samples) ──

_CATEGORY_LABELS = [
    "瓷器", "青铜器", "玉器", "漆器", "金银器",
    "陶器", "雕塑", "纺织品", "珐琅器",
]

_ARTIFACT_TYPE_LABELS = [
    "梅瓶", "玉壶春瓶", "瓶", "壶", "鼎", "簋", "爵", "尊",
    "盘", "碗", "杯", "盏", "罐", "缸",
    "玉璧", "玉琮", "玉璋", "玉圭", "玉璜",
    "俑", "佛像", "镜",
]

_PERIOD_LABELS = [
    "商代", "西周", "春秋", "战国", "秦", "西汉", "东汉",
    "隋", "唐", "五代", "宋", "辽", "金", "元", "明", "清",
]

_MATERIAL_LABELS = [
    "瓷", "青铜", "玉", "漆", "金银", "陶", "石", "木",
]

# Text templates for ensemble prompting (applied per label)
_PROMPT_TEMPLATES = [
    "一张{label}的图片",
    "一件{label}文物",
    "这是{label}",
    "{label}的摄影",
]


class ClipZeroShotBaseline:
    """CLIP zero-shot classification for artifact identification.

    Uses frozen CLIP model and frozen label prompts. Never sees test labels.

    NOTE: This is a STUB implementation. Full functionality requires
    torch and transformers. The run() method will raise ImportError
    if dependencies are missing.
    """

    name = "clip_zero_shot"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        _check_deps()
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._label_cache: dict = {}

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Stub — raises NotImplementedError until full implementation."""
        raise NotImplementedError(
            "CLIP Zero-Shot is not fully implemented in this phase. "
            "Use mock tests or implement with: pip install torch transformers"
        )

    def _lazy_load_model(self) -> None:
        """Load CLIP model on first use."""
        import torch
        import transformers

        self._processor = transformers.CLIPProcessor.from_pretrained(self.model_name)
        self._model = transformers.CLIPModel.from_pretrained(self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
