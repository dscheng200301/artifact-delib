"""Self-Consistency VLM Baseline (external baseline).

Image → Independent VLM Samples × N → Aggregation → Final Prediction

Samples multiple independent VLM responses at temperature > 0, then
aggregates by majority voting. Uses the same underlying VLM as ArtifactDelib.

It answers: Is ArtifactDelib's improvement simply from more model calls?

.. note on seed::
    The seed parameter only controls the sampling order (loop iteration index)
    and does **not** affect API-side stochasticity unless the underlying model
    provider supports and is configured to accept a ``seed`` parameter.
    For fully reproducible results, ensure the provider is configured with a
    fixed seed where supported (e.g., OpenAI's ``seed`` parameter).
"""

from __future__ import annotations

import json
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
    """Image -> N independent VLM samples -> majority aggregation -> prediction.

    Uses structured majority voting on parsed predictions with tie-breaking:

    1. Majority vote on (category, type, period, material) tuple.
    2. If tied on the full tuple, vote on individual fields independently.
    3. If still tied, call a judge VLM to arbitrate among the samples.

    .. note on seed::
        The seed parameter only controls the sampling order (loop iteration
        index) and does **not** affect API-side stochasticity unless the
        underlying model provider supports and is configured to accept a
        ``seed`` parameter.  For fully reproducible results, ensure the
        provider is configured with a fixed seed where supported (e.g.,
        OpenAI's ``seed`` parameter).
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
        self._samples: list[str] = []

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

        # Store raw samples for transparency / downstream inspection
        self._samples = list(samples)

        # ── Aggregation: majority voting on parsed types ──
        aggregated_text, judge_input, judge_output = self._aggregate(samples, encoded)
        total_input += judge_input
        total_output += judge_output

        total_usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)

        # Build sample details as JSON for the summarized report
        sample_details = {
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "seed": self.seed,
            "aggregation_method": "majority_voting_full_tuple_with_per_field_fallback",
            "judge_called": judge_input > 0,
            "samples": [
                {"index": i, "text": s}
                for i, s in enumerate(samples)
            ],
        }
        summary_content = json.dumps(sample_details, ensure_ascii=False, indent=2)

        final = FinalIdentification(content=aggregated_text, usage=total_usage)
        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(content=summary_content, usage=TokenUsage()),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=total_usage,
            total_api_calls=self.n_samples + (1 if judge_input > 0 else 0),
            status="COMPLETED",
        )

    def _aggregate(
        self,
        samples: list[str],
        encoded: str,
    ) -> tuple[str, int, int]:
        """Aggregate N samples using structured majority voting with tie-breaking.

        1. Parse each sample into structured fields.
        2. Majority vote on (category, type, period, material) tuple.
        3. If tied, vote on individual fields independently.
        4. If still tied, call a judge VLM to arbitrate.

        Returns:
            (aggregated_text, extra_input_tokens, extra_output_tokens) where
            ``extra_*_tokens`` are non-zero only when a judge call was made.
        """
        parsed_samples = []
        for text in samples:
            try:
                parsed = self._parser.parse(text)
                parsed_samples.append(parsed)
            except Exception:
                continue

        if not parsed_samples:
            return (samples[0] if samples else ""), 0, 0

        # ── Step 1: majority vote on (category, fine_grained_type, period, material) ──
        tuples = [
            (p.category, p.fine_grained_type, p.period, p.material)
            for p in parsed_samples
        ]
        counter = Counter(tuples)
        top = counter.most_common()

        winner_tuple, winner_count = top[0]
        is_tie = len(top) > 1 and top[0][1] == top[1][1]

        if not is_tie:
            # Clear winner on full tuple
            winner_category, winner_type, winner_period, winner_material = winner_tuple
            result = self._build_result(
                winner_category, winner_type, winner_period, winner_material,
            )
            if result:
                return result, 0, 0
            # If all fields in the winning tuple are None, fall through to per-field

        # ── Step 2: per-field independent majority voting ──
        field_results: dict[str, str | None] = {}
        for field_name, idx in [
            ("category", 0), ("type", 1), ("period", 2), ("material", 3),
        ]:
            field_values = [t[idx] for t in tuples if t[idx] is not None]
            if not field_values:
                field_results[field_name] = None
                continue
            field_counter = Counter(field_values)
            field_top = field_counter.most_common()
            if len(field_top) > 1 and field_top[0][1] == field_top[1][1]:
                # Tie — cannot decide this field
                field_results[field_name] = None
            else:
                field_results[field_name] = field_top[0][0]

        decided = {k: v for k, v in field_results.items() if v is not None}
        if decided:
            result = self._build_result(
                decided.get("category"),
                decided.get("type"),
                decided.get("period"),
                decided.get("material"),
            )
            if result:
                return result, 0, 0

        # ── Step 3: complete tie — call a judge VLM to arbitrate ──
        judge_text, judge_usage = self._call_tie_breaker_judge(samples, encoded)
        return judge_text, judge_usage.input_tokens, judge_usage.output_tokens

    def _call_tie_breaker_judge(
        self,
        samples: list[str],
        encoded: str,
    ) -> tuple[str, TokenUsage]:
        """Call a judge VLM to reconcile conflicting samples when voting ties."""
        sample_list = "\n\n".join(
            f"样本 {i + 1}:\n{s[:400]}"
            for i, s in enumerate(samples)
        )
        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                "你是古代文物鉴定首席专家。多位鉴定师给出了以下不同意见，"
                "请综合分析后做出一致性裁决。\n\n"
                "请输出最终鉴定结论，包括：\n"
                "- 文物大类\n- 具体类型\n- 可能的年代或朝代\n- 可能的材质"
            ),
            user_prompt=(
                f"以下是对同一件文物的 {len(samples)} 个独立鉴定意见，"
                f"但意见存在分歧：\n\n{sample_list}\n\n"
                "请综合分析这些意见，给出最终鉴定结论。"
            ),
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=512,
            temperature=0.3,
            prompt_name="self_consistency_judge",
        ))
        return response.content.strip(), response.usage

    @staticmethod
    def _build_result(
        category: str | None,
        ftype: str | None,
        period: str | None,
        material: str | None,
    ) -> str | None:
        """Build a natural-language result string from structured fields.

        Returns None when all fields are None (nothing to report).
        """
        parts: list[str] = []
        if category:
            parts.append(f"文物大类：{category}")
        if ftype:
            parts.append(f"具体类型：{ftype}")
        if period:
            parts.append(f"年代时期：{period}")
        if material:
            parts.append(f"材质：{material}")
        if parts:
            return "；".join(parts)
        return None

    @property
    def _samples_summary(self) -> str:
        """Return a summary of all samples for logging."""
        return f"self_consistency(n={self.n_samples}, t={self.temperature})"
