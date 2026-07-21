"""BLIP-2 Zero-Shot Baseline — external baseline.

Image → BLIP-2 → Natural-Language Identification → Shared PredictionParser → Prediction

Uses a single BLIP-2 call for NL identification, then parses with the shared
PredictionParser. No expert decomposition, no routing, no candidate analysis.

Requires optional dependencies: torch, transformers, Pillow.
Model weights are NOT downloaded automatically unless allow_model_download=True.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import TokenUsage
from artifact_delib.evaluation.prediction_parser import PredictionParser
from artifact_delib.schemas import (
    CandidateSet,
    FinalIdentification,
    PipelineResult,
    SummarizedReport,
    VisualPerceptionReport,
)

logger = logging.getLogger(__name__)

# Prompt for BLIP-2 — requires 4 output dimensions
BLIP2_PROMPT = (
    "Question: Identify the ancient artifact in this image. "
    "Provide: 1. artifact category (e.g., ceramic, bronze, jade, lacquer), "
    "2. specific artifact type (e.g., vase, tripod, bi-disc), "
    "3. likely historical period or dynasty, "
    "4. material.\n"
    "Answer: This artifact is"
)


class Blip2ZeroShotBaseline:
    """BLIP-2 image-to-text identification baseline.

    Uses a single BLIP-2 call with a structured prompt. The NL output is
    parsed by PredictionParser — the same parser used by all methods.

    Attributes:
        name: Canonical baseline name.
        model_name: HuggingFace model id.
        model_revision: Specific revision.
    """

    name = "blip2_zero_shot"

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        model_revision: str = "main",
        cache_dir: Path | None = None,
        allow_model_download: bool = False,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_dir = cache_dir or Path("data/blip2_cache")
        self.allow_model_download = allow_model_download
        self.device = device

        self._model = None
        self._processor = None
        self._parser = PredictionParser()

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run BLIP-2 zero-shot on a single image."""
        if self._model is None:
            self._load_model()

        t0 = time.perf_counter()

        # Generate NL identification
        nl_text = self._generate(image_path)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Parse with shared parser
        parsed = self._parser.parse(nl_text)

        # Build output
        parts = []
        if parsed.category:
            parts.append(f"文物大类：{parsed.category}")
        if parsed.fine_grained_type:
            parts.append(f"具体类型：{parsed.fine_grained_type}")
        if parsed.period:
            parts.append(f"年代时期：{parsed.period}")
        if parsed.material:
            parts.append(f"材质：{parsed.material}")

        content = "；".join(parts) if parts else nl_text[:200]
        final = FinalIdentification(content=content, usage=TokenUsage())

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(
                content=json.dumps({
                    "model": self.model_name,
                    "revision": self.model_revision,
                    "raw_response": nl_text[:500],
                    "parsed_category": parsed.category,
                    "parsed_type": parsed.fine_grained_type,
                    "parsed_period": parsed.period,
                    "parsed_material": parsed.material,
                }, ensure_ascii=False),
                usage=TokenUsage(),
            ),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=TokenUsage(),
            total_api_calls=0,
            status="COMPLETED",
        )

    def _generate(self, image_path: Path) -> str:
        """Generate NL identification with BLIP-2."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=BLIP2_PROMPT, return_tensors="pt")
        if self.device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=128)
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        return text[0].strip() if text else ""

    def _load_model(self) -> None:
        """Lazy-load BLIP-2. Fail closed if download not allowed."""
        if not self.allow_model_download:
            cache_marker = self.cache_dir / "blip2_model_loaded.txt"
            if not cache_marker.exists():
                raise RuntimeError(
                    "BLIP-2 model download not allowed. "
                    "Set allow_model_download=True or pre-download the model."
                )

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("BLIP-2 requires PyTorch. Install: pip install torch")

        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "BLIP-2 requires transformers. Install: pip install transformers"
            )

        self._processor = Blip2Processor.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model.eval()

        if self.device != "cpu" and torch.cuda.is_available():
            self._model = self._model.to(self.device)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "blip2_model_loaded.txt").touch()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


def _check_deps() -> None:
    """Legacy dependency check (kept for backward compat)."""
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
            "BLIP-2 baseline requires: " + ", ".join(missing) + ". "
            "Install with: pip install torch transformers Pillow"
        )
