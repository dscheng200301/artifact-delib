"""Backward-compat re-exports for the old ablations module.

Previously:
  from artifact_delib.ablations import AblationNoMultiExpert

Still works — it redirects to the new package or provides the original classes.
"""

from __future__ import annotations

import uuid as _uuid
from pathlib import Path as _Path

from artifact_delib.api.base import ModelClient as _ModelClient
from artifact_delib.api.schemas import ModelRequest as _ModelRequest
from artifact_delib.api.schemas import TokenUsage as _TokenUsage

from artifact_delib.agents.artifact_judge import ArtifactJudge as _ArtifactJudge
from artifact_delib.agents.candidate_generator import CandidateGenerator as _CandidateGenerator
from artifact_delib.agents.deliberation.critic_agent import CriticAgent as _CriticAgent
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager as _DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent as _HypothesisAgent
from artifact_delib.agents.disagreement_analyzer import DisagreementAnalyzer as _DisagreementAnalyzer
from artifact_delib.agents.experts.glyph_expert import GlyphExpert as _GlyphExpert
from artifact_delib.agents.experts.local_detail_expert import LocalDetailExpert as _LocalDetailExpert
from artifact_delib.agents.experts.material_craft_expert import MaterialCraftExpert as _MaterialCraftExpert
from artifact_delib.agents.experts.shape_expert import ShapeExpert as _ShapeExpert
from artifact_delib.agents.experts.style_expert import StyleExpert as _StyleExpert
from artifact_delib.agents.report_summarizer import ReportSummarizer as _ReportSummarizer
from artifact_delib.agents.targeted_recheck import TargetedExpertRecheck as _TargetedExpertRecheck
from artifact_delib.agents.visual_perception_agent import VisualPerceptionAgent as _VisualPerceptionAgent
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline as _ArtifactDelibPipeline
from artifact_delib.schemas import (
    CandidateSet as _CandidateSet,
    ExpertReport as _ExpertReport,
    FinalIdentification as _FinalIdentification,
    PipelineResult as _PipelineResult,
    RouteDecision as _RouteDecision,
    SummarizedReport as _SummarizedReport,
    VisualPerceptionReport as _VisualPerceptionReport,
)


def _make_experts(client, model_name):
    return (
        _ShapeExpert(client, model_name),
        _StyleExpert(client, model_name),
        _GlyphExpert(client, model_name),
        _MaterialCraftExpert(client, model_name),
        _LocalDetailExpert(client, model_name),
    )


# ═══════════════════════════════════════════════════════════════
#  New ablations (alias)
# ═══════════════════════════════════════════════════════════════

try:
    from artifact_delib.ablations.no_expert_specialization import AblationNoExpertSpecialization
    from artifact_delib.ablations.no_disagreement_analysis import AblationNoDisagreementAnalysis
    from artifact_delib.ablations.no_dynamic_routing import AblationNoDynamicRouting
    from artifact_delib.ablations.random_recheck import AblationRandomRecheck
    from artifact_delib.ablations.no_controlled_deliberation import AblationNoControlledDeliberation
    from artifact_delib.ablations.free_debate import AblationFreeDebateNew
    from artifact_delib.ablations.no_critic import AblationNoCritic
    from artifact_delib.ablations.early_judge import AblationEarlyJudge
    _HAS_NEW_ABLATIONS = True
except ImportError:
    _HAS_NEW_ABLATIONS = False


# ═══════════════════════════════════════════════════════════════
#  A1: w/o Multi-Expert — skip all 5 experts (legacy, preserved)
# ═══════════════════════════════════════════════════════════════

class AblationNoMultiExpert(_ArtifactDelibPipeline):
    """A1 (legacy): VP → Judge directly. No expert analysis."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
        total_calls = 0
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1
        empty_summary = _SummarizedReport(content=vp.content, usage=_TokenUsage())
        empty_candidates = _CandidateSet(candidates=())
        final = self.judge.adjudicate(image_path, vp, empty_summary, empty_candidates, ())
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=(),
            summarized_report=empty_summary, initial_candidates=empty_candidates,
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A2: Single General Expert (legacy)
# ═══════════════════════════════════════════════════════════════

class AblationSingleExpert(_ArtifactDelibPipeline):
    """A2 (legacy): VP → Single Generic Expert → Summarizer → Judge."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
        import base64
        total_calls = 0
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self.client.generate(_ModelRequest(
            request_id=str(_uuid.uuid4()), model=self.model_name,
            system_prompt="你是古代文物全科鉴定专家。请对这张文物图片进行全面分析。\n\n覆盖以下维度：\n- 器形特征\n- 纹饰与艺术风格\n- 铭文款识\n- 材质与工艺\n- 局部细节\n\n输出一段200-400字的综合专业分析。不要输出JSON。",
            user_prompt="请对这件文物进行全面分析。",
            image_base64=f"data:image/png;base64,{encoded}",
            max_output_tokens=512, temperature=0.0, prompt_name="generic_expert",
        ))
        total_calls += 1
        single_report = _ExpertReport(expert_name="general", content=response.content.strip(), usage=response.usage)
        summary = self.summarizer.summarize(vp, (single_report,))
        total_calls += 1
        candidates = self.candidate_generator.generate(summary)
        total_calls += 1
        final = self.judge.adjudicate(image_path, vp, summary, candidates, (single_report,))
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=(single_report,),
            summarized_report=summary, initial_candidates=candidates,
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A3: w/o Dynamic Router (legacy)
# ═══════════════════════════════════════════════════════════════

class AblationNoRouter(_ArtifactDelibPipeline):
    """A3 (legacy): VP → 5 Experts → Summarizer → Candidates → Judge. No router."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
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
        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


# ═══════════════════════════════════════════════════════════════
#  A4, A5, A8, A9, A10, A12: remaining legacy variants
# ═══════════════════════════════════════════════════════════════

class AblationNoRecheck(_ArtifactDelibPipeline):
    """A4 (legacy): Skip recheck, go to Judge."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
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
        final = self.judge.adjudicate(image_path, vp, summary, candidates, et)
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            disagreement_analysis=disagreement,
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


class AblationFixedAllRecheck(_ArtifactDelibPipeline):
    """A5 (legacy): Always run all 5 recheck types."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
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
        recheck_records = []
        for action in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                       "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK"):
            route = _RouteDecision(action=action, reason="fixed_all", recheck_count=0)
            record = self.recheck_coordinator.execute(
                image_path, route, candidates, disagreement, et,
                recheck_history=tuple(recheck_records), round_no=len(recheck_records) + 1,
            )
            recheck_records.append(record)
            total_calls += 1
        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, et,
            recheck_reports=tuple(
                _ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
        )
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary, initial_candidates=candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(
                _RouteDecision(action=a, reason="fixed_all", recheck_count=0)
                for a in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                          "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK")
            ),
            recheck_reports=tuple(
                _ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


class AblationNoDeliberation(_ArtifactDelibPipeline):
    """A8 (legacy): Skip deliberation."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
        result = super().run(image_path, sample_id)
        if result.deliberation_result and result.deliberation_result.rounds:
            result = _PipelineResult(
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


class AblationFreeDebate(_ArtifactDelibPipeline):
    """A9 (legacy): Free debate instead of controlled deliberation."""

    def __init__(self, client, model_name="default", **kwargs):
        super().__init__(client, model_name, **kwargs)
        from artifact_delib.baselines import GenericMADBaseline
        self._debate = GenericMADBaseline(client, model_name, n_agents=2, n_rounds=2)

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
        result = super().run(image_path, sample_id)
        if result.deliberation_result:
            debate_result = self._debate.run(image_path, sample_id)
            result = _PipelineResult(
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


class AblationFixedDeliberation(_ArtifactDelibPipeline):
    """A10 (legacy): Always run deliberation."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
        result = super().run(image_path, sample_id)
        if result.deliberation_result is None:
            delib_result = self.deliberation_manager.deliberate(
                candidates=result.initial_candidates,
                summarized_report=result.summarized_report,
                expert_reports=result.expert_reports,
                recheck_reports=result.recheck_reports,
            )
            result = _PipelineResult(
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


class AblationNoDeferredJudge(_ArtifactDelibPipeline):
    """A12 (legacy): Judge before candidate generation."""

    def run(self, image_path: _Path, sample_id: str = "unknown") -> _PipelineResult:
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
        early_final = self.judge.adjudicate(
            image_path, vp, summary, _CandidateSet(candidates=()), et,
        )
        total_calls += 1
        return _PipelineResult(
            sample_id=sample_id, final_identification=early_final,
            visual_perception_report=vp, expert_reports=et,
            summarized_report=summary,
            initial_candidates=_CandidateSet(candidates=()),
            total_usage=_TokenUsage(), total_api_calls=total_calls, status="COMPLETED",
        )


__all__ = [
    "AblationNoMultiExpert",
    "AblationSingleExpert",
    "AblationNoRouter",
    "AblationNoRecheck",
    "AblationFixedAllRecheck",
    "AblationNoDeliberation",
    "AblationFreeDebate",
    "AblationFixedDeliberation",
    "AblationNoDeferredJudge",
]
