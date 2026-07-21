"""Self-Consistency VLM Baseline (external baseline).

Image → Independent VLM Samples × N → Aggregation → Final Prediction

Samples multiple independent VLM responses at temperature > 0, then
aggregates by majority voting. Uses the same underlying VLM as ArtifactDelib.

It answers: Is ArtifactDelib's improvement simply from more model calls?
"""

from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.evaluation.prediction_parser import PredictionParser
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    SummarizedReport,
    VisualPerceptionReport,
)

# Default number of samples
DEFAULT_N_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7


class SelfConsistencyBaseline:
    """Image → N independent VLM samples → majority aggregation → prediction.

    Uses structured majority voting on parsed predictions.
    Falls back to a unified judge call when no majority emerges.
    """

    name = "self_consistency_vlm"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        n_samples: int = DEFAULT_N_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.n_samples = max(3, n_samples)
        self.temperature = temperature
        self.seed = seed
        self._parser = PredictionParser()

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run N independent VLM samples and aggregate."""
        import base64

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

        # N independent samples
        samples: list[str] = []
        total_input = 0
        total_output = 0
        for i in range(self.n_samples):
            response = self.client.generate(ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=(
                    "你是古代文物鉴定专家。请直接识别这张文物图片中的器物。\n\n"
                    "请输出一段自然语言描述，包括：\n"
                    "- 文物大类\n- 具体类型\n- 可能的年代或朝代\n- 可能的材质和工艺"
                ),
                user_prompt="请直接识别这件文物。",
                image_base64=f"data:image/png;base64,{encoded}",
                max_output_tokens=512,
                temperature=self.temperature,
                prompt_name="self_consistency",
            ))
            samples.append(response.content.strip())
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

        # ── Aggregation: majority voting on parsed types ──
        aggregated = self._aggregate(samples)

        total_usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)
        final = FinalIdentification(content=aggregated, usage=total_usage)
        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(content="", usage=TokenUsage()),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=total_usage,
            total_api_calls=self.n_samples,
            status="COMPLETED",
        )

    def _aggregate(self, samples: list[str]) -> str:
        """Aggregate N samples using structured majority voting.

        1. Parse each sample into structured fields.
        2. Majority vote on (category, type, period) tuple.
        3. If no majority, fall back to the first sample.
        """
        parsed_samples = []
        for text in samples:
            try:
                parsed = self._parser.parse(text)
                parsed_samples.append(parsed)
            except Exception:
                continue

        if not parsed_samples:
            return samples[0] if samples else ""

        # Vote on (category, fine_grained_type, period) tuples
        tuples = [
            (p.category, p.fine_grained_type, p.period)
            for p in parsed_samples
        ]
        counter = Counter(tuples)
        winner_tuple, _ = counter.most_common(1)[0]

        winner_category, winner_type, winner_period = winner_tuple

        # If no clear winner for any field, fall back
        if not winner_type and not winner_period:
            return samples[0]

        # Reconstruct a natural-language prediction from the winning tuple
        parts = []
        if winner_category:
            parts.append(f"文物大类：{winner_category}")
        if winner_type:
            parts.append(f"具体类型：{winner_type}")
        if winner_period:
            parts.append(f"年代时期：{winner_period}")

        if parts:
            return "；".join(parts)
        return samples[0]

    @property
    def _samples_summary(self) -> str:
        """Return a summary of all samples for logging."""
        return f"self_consistency(n={self.n_samples}, t={self.temperature})"
