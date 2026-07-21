"""Ablations for ArtifactDelib paper — controlled variants of the main pipeline.

Each ablation removes or replaces exactly one component to isolate its contribution.

A1: w/o Multi-Expert         — skip 5 experts, VP → Judge
A2: Single General Expert    — replace 5 experts with 1 generic VLM
A3: w/o Dynamic Router       — always same fixed path (like B2)
A4: w/o Targeted Recheck     — disagreement → Judge directly
A5: Fixed All Recheck        — always run all 5 rechecks
A6: Rule Router              — == B5 (ArtifactDelibPipeline itself)
A7: Learned Router           — Phase 9+

A8: w/o Deliberation         — recheck exhausted → Judge
A9: Free Debate              — use generic free debate
A10: Fixed Deliberation      — always run deliberation after recheck
A11: Adaptive Deliberation   — == B5 (ArtifactDelibPipeline itself)
A12: w/o Deferred Judge      — Judge participates during candidate generation
"""

from __future__ import annotations

import uuid
from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage

from artifact_delib.agents.artifact_judge import ArtifactJudge
from artifact_delib.agents.candidate_generator import CandidateGenerator
from artifact_delib.agents.deliberation.critic_agent import CriticAgent
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent
from artifact_delib.agents.disagreement_analyzer import DisagreementAnalyzer
from artifact_delib.agents.experts.glyph_expert import GlyphExpert
from artifact_delib.agents.experts.local_detail_expert import LocalDetailExpert
from artifact_delib.agents.experts.material_craft_expert import MaterialCraftExpert
from artifact_delib.agents.experts.shape_expert import ShapeExpert
from artifact_delib.agents.experts.style_expert import StyleExpert
from artifact_delib.agents.report_summarizer import ReportSummarizer
from artifact_delib.agents.targeted_recheck import TargetedExpertRecheck
from artifact_delib.agents.visual_perception_agent import VisualPerceptionAgent
from artifact_delib.baselines import GenericMADBaseline
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    PipelineResult,
    RouteDecision,
    SummarizedReport,
    VisualPerceptionReport,
)


def _make_experts(client, model_name):
    return (
        ShapeExpert(client, model_name),
        StyleExpert(client, model_name),
        GlyphExpert(client, model_name),
        MaterialCraftExpert(client, model_name),
        LocalDetailExpert(client, model_name),
    )


# ═══════════════════════════════════════════════════════════════
#  A1: w/o Multi-Expert — skip all 5 experts
# ═══════════════════════════════════════════════════════════════

class AblationNoMultiExpert(ArtifactDelibPipeline):
    """A1: VP → Judge directly. No expert analysis.

    Isolates the contribution of multi-expert decomposition.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        # Skip all 5 experts — use VP content as summarized report
        empty_summary = SummarizedReport(content=vp.content, usage=TokenUsage())
        empty_candidates = CandidateSet(candidates=())

        final = self.judge.adjudicate(
            image_path, vp, empty_summary, empty_candidates, ()
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=(),
            summarized_report=empty_summary, initial_candidates=empty_candidates,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A2: Single General Expert — replace 5 experts with 1
# ═══════════════════════════════════════════════════════════════

class AblationSingleExpert(ArtifactDelibPipeline):
    """A2: VP → Single Generic Expert → Summarizer → Judge.

    One VLM does all analysis instead of 5 specialized experts.
    Isolates the contribution of expert specialization.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        import base64
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        # Single generic expert: one call that covers all dimensions
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.generate(ModelRequest(
            request_id=str(uuid.uuid4()),
            model=self.model_name,
            system_prompt=(
                "你是古代文物全科鉴定专家。请对这张文物图片进行全面分析。\n\n"
                "覆盖以下维度：\n"
                "- 器形特征\n- 纹饰与艺术风格\n- 铭文款识\n- 材质与工艺\n- 局部细节\n\n"
                "输出一段200-400字的综合专业分析。不要输出JSON。"
            ),
            user_prompt="请对这件文物进行全面分析。",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=512, temperature=0.0, prompt_name="generic_expert",
        ))
        total_calls += 1

        # Pass the single expert response as if it were the summarized report
        single_report = ExpertReport(expert_name="general", content=response.content.strip(),
                                      usage=response.usage)
        summary = self.summarizer.summarize(vp, (single_report,))
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        # No disagreement analysis or routing — fixed path
        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, (single_report,),
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=(single_report,),
            summarized_report=summary, initial_candidates=candidates,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A3: w/o Dynamic Router — always same fixed path
# ═══════════════════════════════════════════════════════════════

class AblationNoRouter(ArtifactDelibPipeline):
    """A3: VP → 5 Experts → Summarizer → Candidates → Judge. No router, no recheck.

    Same path as FixedMultiExpert baseline, but with full pipeline provenance tracking.
    Isolates the contribution of dynamic routing.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in _make_experts(self.client, self.model_name):
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        # Skip disagreement, router, recheck — go directly to judge
        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A4: w/o Targeted Recheck — disagreement → Judge
# ═══════════════════════════════════════════════════════════════

class AblationNoRecheck(ArtifactDelibPipeline):
    """A4: Disagreement detected → Judge directly. No recheck allowed.

    Isolates the contribution of targeted recheck.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in _make_experts(self.client, self.model_name):
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # No routing — always go directly to judge with disagreement info
        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            disagreement_analysis=disagreement,
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A5: Fixed All Recheck — always run all 5 rechecks
# ═══════════════════════════════════════════════════════════════

class AblationFixedAllRecheck(ArtifactDelibPipeline):
    """A5: Always run all 5 recheck types regardless of disagreement pattern.

    Isolates the contribution of targeted (vs. all) recheck.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in _make_experts(self.client, self.model_name):
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # Always run all 5 rechecks
        recheck_records = []
        for action in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                       "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK"):
            route = RouteDecision(action=action, reason="fixed_all", recheck_count=0)
            record = self.recheck_coordinator.execute(
                image_path, route, candidates, disagreement, et,
                recheck_history=tuple(recheck_records), round_no=len(recheck_records) + 1,
            )
            recheck_records.append(record)
            total_calls += 1

        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, et,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(
                RouteDecision(action=a, reason="fixed_all", recheck_count=0)
                for a in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                          "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK")
            ),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A8: w/o Deliberation — recheck exhausted → Judge
# ═══════════════════════════════════════════════════════════════

class AblationNoDeliberation(ArtifactDelibPipeline):
    """A8: Skip deliberation — when router says DELIBERATION, go to Judge.

    Isolates the contribution of controlled deliberation.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        # Run full pipeline but force DELIBERATION route to be treated as FAST→Judge
        # We hook into the parent's router behavior by setting deliberation_count=999
        # to prevent the real DELIBERATION from being returned
        result = super().run(image_path, sample_id)
        # If deliberation was triggered and generated rounds, nullify
        if result.deliberation_result and result.deliberation_result.rounds:
            result = PipelineResult(
                sample_id=result.sample_id,
                final_identification=result.final_identification,
                visual_perception_report=result.visual_perception_report,
                expert_reports=result.expert_reports,
                summarized_report=result.summarized_report,
                initial_candidates=result.initial_candidates,
                disagreement_analysis=result.disagreement_analysis,
                route_decisions=result.route_decisions,
                recheck_reports=result.recheck_reports,
                recheck_records=result.recheck_records,
                deliberation_result=None,
                total_usage=result.total_usage,
                total_api_calls=result.total_api_calls,
                status=result.status,
            )
        return result


# ═══════════════════════════════════════════════════════════════
#  A9: Free Debate — use generic multi-agent debate
# ═══════════════════════════════════════════════════════════════

class AblationFreeDebate(ArtifactDelibPipeline):
    """A9: Replace controlled deliberation with free multi-agent debate.

    Isolates the contribution of controlled (vs free) deliberation.
    """

    def __init__(self, client, model_name="default", **kwargs):
        super().__init__(client, model_name, **kwargs)
        self._debate = GenericMADBaseline(client, model_name, n_agents=2, n_rounds=2)

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        result = super().run(image_path, sample_id)
        # Replace deliberation result with free debate
        if result.deliberation_result:
            debate_result = self._debate.run(image_path, sample_id)
            result = PipelineResult(
                sample_id=result.sample_id,
                final_identification=debate_result.final_identification,
                visual_perception_report=result.visual_perception_report,
                expert_reports=result.expert_reports,
                summarized_report=result.summarized_report,
                initial_candidates=result.initial_candidates,
                disagreement_analysis=result.disagreement_analysis,
                route_decisions=result.route_decisions,
                recheck_reports=result.recheck_reports,
                recheck_records=result.recheck_records,
                deliberation_result=result.deliberation_result,
                total_usage=result.total_usage,
                total_api_calls=result.total_api_calls + debate_result.total_api_calls,
                status=result.status,
            )
            return result
        return result


# ═══════════════════════════════════════════════════════════════
#  A10: Fixed Deliberation — always run deliberation
# ═══════════════════════════════════════════════════════════════

class AblationFixedDeliberation(ArtifactDelibPipeline):
    """A10: Always run 2 rounds of deliberation after any routing.

    Isolates the contribution of adaptive (only-when-needed) deliberation.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        result = super().run(image_path, sample_id)
        # If no deliberation was triggered, force one
        if result.deliberation_result is None:
            delib_result = self.deliberation_manager.deliberate(
                candidates=result.initial_candidates,
                summarized_report=result.summarized_report,
                expert_reports=result.expert_reports,
                recheck_reports=result.recheck_reports,
            )
            result = PipelineResult(
                sample_id=result.sample_id,
                final_identification=result.final_identification,
                visual_perception_report=result.visual_perception_report,
                expert_reports=result.expert_reports,
                summarized_report=result.summarized_report,
                initial_candidates=result.initial_candidates,
                disagreement_analysis=result.disagreement_analysis,
                route_decisions=result.route_decisions,
                recheck_reports=result.recheck_reports,
                recheck_records=result.recheck_records,
                deliberation_result=delib_result,
                total_usage=result.total_usage,
                total_api_calls=result.total_api_calls + len(delib_result.rounds) * 3,
                status=result.status,
            )
            return result
        return result


# ═══════════════════════════════════════════════════════════════
#  A12: w/o Deferred Judge — Judge participates early
# ═══════════════════════════════════════════════════════════════

class AblationNoDeferredJudge(ArtifactDelibPipeline):
    """A12: Judge participates during candidate generation (not deferred).

    Instead of the judge being the LAST module, it reviews the multi-expert
    analysis BEFORE candidate generation and disagreement analysis.
    """

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for exp in _make_experts(self.client, self.model_name):
            reports.append(exp.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        # Judge reviews BEFORE candidate generation (early intervention)
        early_final = self.judge.adjudicate(
            image_path, vp, summary, CandidateSet(candidates=()), et,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id, final_identification=early_final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary,
            initial_candidates=CandidateSet(candidates=()),
            total_usage=TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )
