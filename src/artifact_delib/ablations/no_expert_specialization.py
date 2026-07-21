"""A1: w/o Expert Specialization.

Replace 5 specialized experts with 5 identical general experts.
Same number of agents, same API calls, but all use the same generic prompt.

This isolates whether improvement is from specialization or from
simply making multiple calls.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage
from artifact_delib.constants import DEFAULT_TOP_K, MAX_DELIBERATION_ROUNDS, MAX_RECHECK_ROUNDS
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import (
    ExpertReport,
    PipelineResult,
    RecheckRecord,
    RouteDecision,
)


# Generic expert prompt — same for all 5 "experts"
_GENERIC_EXPERT_PROMPT = (
    "你是古代文物鉴定专家。请从全面角度分析这张文物图片，覆盖以下方面：\n"
    "- 器形特征\n- 纹饰风格\n- 文字/铭文/款识\n- 材质与工艺\n- 局部细节\n\n"
    "输出200-400字的综合专业分析。不要输出JSON。"
)


class AblationNoExpertSpecialization(ArtifactDelibPipeline):
    """A1: Five identical general experts instead of five specialized ones.

    Keeps the same pipeline structure (VP, 5 agents, summarizer, candidates,
    disagreement, router, recheck, deliberation, judge) but uses the same
    generic prompt for all 5 "expert" agents.
    """

    name = "ablation_no_expert_specialization"

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run pipeline with 5 generic agents instead of specialized experts."""
        import base64

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        total_calls = 0

        # Step 1: VP
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        # Step 2: 5 identical generic experts
        expert_reports = []
        for _ in range(5):
            response = self.client.generate(ModelRequest(
                request_id=str(uuid.uuid4()),
                model=self.model_name,
                system_prompt=_GENERIC_EXPERT_PROMPT,
                user_prompt="请全面分析这件文物。",
                image_base64=f"data:image/png;base64,{encoded}",
                max_output_tokens=512,
                temperature=0.3,
                prompt_name="generic_expert",
            ))
            expert_reports.append(ExpertReport(
                expert_name="general",
                content=response.content.strip(),
                usage=response.usage,
            ))
            total_calls += 1

        et = tuple(expert_reports)

        # Step 3-4: Summarizer, Candidates
        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        # Step 5: Disagreement
        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # Step 6+: Routing loop (same as full pipeline)
        route_decisions: list[RouteDecision] = []
        recheck_records: list[RecheckRecord] = []
        deliberation_result = None
        recheck_count = 0
        deliberation_count = 0
        completed_rechecks: list[str] = []
        initial_candidates = candidates

        while True:
            route = self.router.route(
                disagreement=disagreement,
                candidates=candidates,
                recheck_count=recheck_count,
                deliberation_count=deliberation_count,
                completed_rechecks=tuple(completed_rechecks),
            )
            route_decisions.append(route)

            if route.action == "FAST":
                break

            if route.action == "DELIBERATION":
                deliberation_count += 1
                deliberation_result = self.deliberation_manager.deliberate(
                    candidates=candidates,
                    summarized_report=summary,
                    expert_reports=et,
                    recheck_reports=tuple(
                        ExpertReport(
                            expert_name=r.expert_name,
                            content=r.new_content,
                            usage=r.usage,
                        ) for r in recheck_records
                    ),
                )
                for _ in deliberation_result.rounds:
                    total_calls += 3
                break

            # Recheck (uses same generic expert)
            if route.action in (
                "SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK",
            ):
                # Call generic expert for recheck
                response = self.client.generate(ModelRequest(
                    request_id=str(uuid.uuid4()),
                    model=self.model_name,
                    system_prompt=_GENERIC_EXPERT_PROMPT + "\n\n请基于以下信息进行定向重审。候选：" + str(candidates),
                    user_prompt="请再次全面分析这件文物，重点关注不确定的方面。",
                    image_base64=f"data:image/png;base64,{encoded}",
                    max_output_tokens=512,
                    temperature=0.3,
                    prompt_name="generic_recheck",
                ))
                total_calls += 1
                recheck_count += 1
                completed_rechecks.append(route.action)

                # Re-summarize, re-generate candidates, re-analyze
                summary = self.summarizer.summarize(vp, et)
                total_calls += 1
                candidates = self.candidate_generator.generate(summary)
                total_calls += 1
                disagreement = self.disagreement_analyzer.analyze(candidates, summary)
                total_calls += 1
                continue

            break

        # Final Judge
        final = self.judge.adjudicate(
            image_path=image_path,
            visual_report=vp,
            summarized_report=summary,
            candidates=candidates,
            expert_reports=et,
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                ) for r in recheck_records
            ),
            deliberation_result=deliberation_result,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=vp,
            expert_reports=et,
            summarized_report=summary,
            initial_candidates=initial_candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(route_decisions),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=TokenUsage(),
            total_api_calls=total_calls,
            status="COMPLETED",
        )
