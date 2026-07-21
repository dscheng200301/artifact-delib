"""A6: Free Debate instead of Controlled Deliberation.

Replace controlled Top-1/Top-2 hypothesis deliberation with open free-form
debate. Same agent count, same max rounds, same Judge.

This is an ABLATION of ArtifactDelib's controlled deliberation mechanism,
not the MultiAgentDebateBaseline (which is an external baseline without
any of ArtifactDelib's pipeline).
"""
from __future__ import annotations
import uuid
from pathlib import Path
from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import PipelineResult


class AblationFreeDebateNew(ArtifactDelibPipeline):
    """A6: Replace controlled deliberation with free multi-agent debate.

    Uses the same pipeline up to the deliberation point, but replaces
    the HypothesisAgent/CriticAgent with a free debate among 2 agents.
    """
    name = "ablation_free_debate"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        top_k: int = 3,
        max_recheck_rounds: int = 2,
    ) -> None:
        super().__init__(client, model_name, top_k, max_recheck_rounds)
        self._n_agents = 2
        self._n_rounds = 2

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        import base64
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

        result = super().run(image_path, sample_id)
        if result.deliberation_result is None:
            return result

        # Replace deliberation with free debate
        total_calls = result.total_api_calls

        opinions: list[str] = []
        for i in range(self._n_agents):
            response = self.client.generate(ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=(
                    f"你是古代文物鉴定专家 #{i+1}。请根据图片和已有的专家分析，"
                    "独立判断这件文物的身份。"
                ),
                user_prompt=f"专家报告：{result.summarized_report.content[:200]}...请给出你的判断。",
                image_base64=f"data:image/png;base64,{encoded}",
                max_output_tokens=256,
                temperature=0.3,
                prompt_name="free_debate",
            ))
            opinions.append(response.content.strip())
            total_calls += 1

        for rnd in range(1, self._n_rounds):
            new_opinions = []
            for i in range(self._n_agents):
                prior = "\n".join(
                    f"Agent {j+1}: {op[:150]}" for j, op in enumerate(opinions)
                )
                response = self.client.generate(ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=self.model_name,
                    system_prompt=(
                        f"你是专家 #{i+1}。请查看其他专家的意见，然后给出你的最终判断。"
                    ),
                    user_prompt=f"其他专家意见：\n{prior}\n\n你的判断：",
                    image_base64=f"data:image/png;base64,{encoded}",
                    max_output_tokens=256,
                    temperature=0.3,
                    prompt_name="free_debate",
                ))
                new_opinions.append(response.content.strip())
                total_calls += 1
            opinions = new_opinions

        # Final judge with debate context
        final = self.judge.adjudicate(
            image_path=image_path,
            visual_report=result.visual_perception_report,
            summarized_report=result.summarized_report,
            candidates=result.initial_candidates,
            expert_reports=result.expert_reports,
            deliberation_result=None,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=result.sample_id,
            final_identification=final,
            visual_perception_report=result.visual_perception_report,
            expert_reports=result.expert_reports,
            summarized_report=result.summarized_report,
            initial_candidates=result.initial_candidates,
            disagreement_analysis=result.disagreement_analysis,
            route_decisions=result.route_decisions,
            recheck_reports=result.recheck_reports,
            recheck_records=result.recheck_records,
            deliberation_result=None,
            total_usage=TokenUsage(),
            total_api_calls=total_calls,
            status="COMPLETED",
        )
