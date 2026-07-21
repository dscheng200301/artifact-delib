"""Direct Single-VLM Baseline (external baseline).

Image → One VLM Call → Final Identification

This baseline uses the same underlying VLM as ArtifactDelib but without:
  - Five specialized experts
  - Candidate generation
  - Disagreement analysis
  - Dynamic routing
  - Deliberation

It answers: Is ArtifactDelib better than a single VLM call with the same backbone?
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    SummarizedReport,
    VisualPerceptionReport,
)


class DirectVLMBaseline:
    """Image → Single VLM → Final Answer.

    A single model call. Proves whether multi-expert decomposition improves accuracy.
    This is the lower-bound baseline for the paper's external baseline table.
    """

    name = "direct_single_vlm"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
    ) -> None:
        self.client = client
        self.model_name = model_name

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run one VLM call and return a PipelineResult-compatible output."""
        import base64

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                "你是古代文物鉴定专家。请直接识别这张文物图片中的器物。\n\n"
                "请输出一段自然语言描述，包括：\n"
                "- 文物大类\n- 具体类型\n- 可能的年代或朝代\n- 可能的材质和工艺\n\n"
                "不要输出JSON。直接输出一段流畅的自然语言。"
            ),
            user_prompt="请直接识别这件文物。",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=512,
            temperature=0.0,
            prompt_name="direct_vlm",
        ))

        final = FinalIdentification(content=response.content.strip(), usage=response.usage)
        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=VisualPerceptionReport(content="", usage=TokenUsage()),
            expert_reports=(),
            summarized_report=SummarizedReport(content="", usage=TokenUsage()),
            initial_candidates=CandidateSet(candidates=()),
            total_usage=response.usage,
            total_api_calls=1,
            status="COMPLETED",
        )
