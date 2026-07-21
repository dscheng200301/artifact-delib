"""ArtifactDelib pipeline with pluggable stages.

The pipeline is decomposed into stage methods that each operate on a shared
PipelineState. Ablation variants override individual stages rather than
reimplementing the entire run() method.

Flow:
  Image → VP → 5 Experts → Summarizer → Top-K → Disagreement →
    Router → [FAST | Recheck → Loop | Deliberation] → Judge
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import TokenUsage

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
from artifact_delib.constants import (
    DEFAULT_TOP_K,
    MAX_DELIBERATION_ROUNDS,
    MAX_RECHECK_ROUNDS,
)
from artifact_delib.pipeline.state import PipelineState, RunAccounting
from artifact_delib.router.rule_router import RuleRouter
from artifact_delib.schemas import (
    ExpertReport,
    PipelineResult,
    RecheckRecord,
    RouteDecision,
)

# Map route action → expert name string
_ACTION_TO_EXPERT_NAME = {
    "SHAPE_RECHECK": "shape",
    "STYLE_RECHECK": "style",
    "GLYPH_RECHECK": "glyph",
    "MATERIAL_RECHECK": "material",
    "LOCAL_DETAIL_RECHECK": "local_detail",
}


class ArtifactDelibPipeline:
    """ArtifactDelib pipeline with pluggable stage methods.

    Each stage is a separate method that reads/writes PipelineState.
    Ablation variants override individual stages for targeted changes.
    """

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        top_k: int = DEFAULT_TOP_K,
        max_recheck_rounds: int = MAX_RECHECK_ROUNDS,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.top_k = top_k
        self.max_recheck_rounds = max(1, max_recheck_rounds)

        # Agents
        self.visual_perception = VisualPerceptionAgent(client, model_name)
        self.shape_expert = ShapeExpert(client, model_name)
        self.style_expert = StyleExpert(client, model_name)
        self.glyph_expert = GlyphExpert(client, model_name)
        self.material_expert = MaterialCraftExpert(client, model_name)
        self.local_detail_expert = LocalDetailExpert(client, model_name)
        self.summarizer = ReportSummarizer(client, model_name)
        self.candidate_generator = CandidateGenerator(client, model_name, top_k)
        self.disagreement_analyzer = DisagreementAnalyzer(client, model_name)
        self.router = RuleRouter(max_recheck_rounds)
        self.recheck_coordinator = TargetedExpertRecheck(
            shape_expert=self.shape_expert,
            style_expert=self.style_expert,
            glyph_expert=self.glyph_expert,
            material_expert=self.material_expert,
            local_detail_expert=self.local_detail_expert,
        )
        self.deliberation_manager = DeliberationManager(
            hypothesis_agent=HypothesisAgent(client, model_name),
            critic_agent=CriticAgent(client, model_name),
            max_rounds=MAX_DELIBERATION_ROUNDS,
        )
        self.judge = ArtifactJudge(client, model_name)

    # ═══════════════════════════════════════════════════
    #  Main orchestration
    # ═══════════════════════════════════════════════════

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run the full pipeline with pluggable stages."""
        state = PipelineState(image_path=image_path, sample_id=sample_id)

        self._stage_initial_analysis(state)
        self._stage_routing_loop(state)
        self._stage_deliberation(state)
        self._stage_final_judge(state)

        return state.to_pipeline_result()

    # ═══════════════════════════════════════════════════
    #  Stage 1: Initial Analysis
    # ═══════════════════════════════════════════════════

    def _stage_initial_analysis(self, state: PipelineState) -> None:
        """VP → experts → summarizer → candidates → disagreement."""
        # VP
        vp = self.visual_perception.analyze(state.image_path)
        state.accounting.record_call("visual_perception", vp.usage)
        state.visual_report = vp

        # Experts
        exp_reports = self._run_experts(state)
        state.expert_reports = exp_reports

        # Summarize
        summary = self.summarizer.summarize(vp, tuple(exp_reports))
        state.accounting.record_call("summarizer", summary.usage)
        state.summarized_report = summary

        # Candidates
        cands = self.candidate_generator.generate(summary)
        state.accounting.record_call("candidate_generator", cands.usage)
        state.initial_candidates = cands
        state.current_candidates = cands

        # Disagreement
        disag = self._analyze_disagreement(state)
        state.disagreement = disag

    def _run_experts(self, state: PipelineState) -> list[ExpertReport]:
        """Run all 5 specialized experts. Override in A1."""
        reports: list[ExpertReport] = []
        for expert in [
            self.shape_expert, self.style_expert, self.glyph_expert,
            self.material_expert, self.local_detail_expert,
        ]:
            r = expert.analyze(state.image_path)
            state.accounting.record_call(r.expert_name, r.usage)
            reports.append(r)
        return reports

    def _analyze_disagreement(
        self, state: PipelineState,
    ) -> DisagreementAnalysis | None:
        """Analyze disagreement. Override in A2 to use margin-only."""
        if state.current_candidates is None:
            return None
        disag = self.disagreement_analyzer.analyze(
            state.current_candidates, state.summarized_report
        )
        state.accounting.record_call("disagreement_analyzer", disag.usage)
        return disag

    # ═══════════════════════════════════════════════════
    #  Stage 2: Routing Loop
    # ═══════════════════════════════════════════════════

    def _stage_routing_loop(self, state: PipelineState) -> None:
        """Route → [FAST | Recheck → Loop | Deliberation]."""
        while True:
            route = self._make_routing_decision(state)
            state.route_decisions.append(route)

            if route.action == "FAST":
                break

            if route.action == "DELIBERATION":
                state.deliberation_count += 1
                break  # Deliberation is handled in _stage_deliberation

            # Targeted Recheck
            if route.action in _ACTION_TO_EXPERT_NAME:
                self._execute_recheck(state, route)
                self._recompute_after_recheck(state)
                continue

            break

    def _make_routing_decision(self, state: PipelineState) -> RouteDecision:
        """Decide next action. Override in A2 (margin-only), A4 (random)."""
        return self.router.route(
            disagreement=state.disagreement,
            candidates=state.current_candidates
            or state.initial_candidates
            or CandidateSet(candidates=()),
            recheck_count=state.recheck_count,
            deliberation_count=state.deliberation_count,
            completed_rechecks=tuple(state.completed_rechecks),
        )

    def _execute_recheck(
        self, state: PipelineState, route: RouteDecision,
    ) -> None:
        """Execute one targeted recheck."""
        record = self.recheck_coordinator.execute(
            image_path=state.image_path,
            route=route,
            candidates=state.current_candidates
            or state.initial_candidates
            or CandidateSet(candidates=()),
            disagreement=state.disagreement,
            current_reports=tuple(state.expert_reports),
            recheck_history=tuple(state.recheck_records),
            round_no=state.recheck_count + 1,
        )
        state.accounting.record_call("recheck", record.usage)
        state.recheck_records.append(record)
        state.recheck_count += 1
        state.completed_rechecks.add(route.action)

        # Update expert reports
        state.expert_reports = _replace_expert_report(
            state.expert_reports,
            _ACTION_TO_EXPERT_NAME[route.action],
            ExpertReport(
                expert_name=_ACTION_TO_EXPERT_NAME[route.action],
                content=record.new_content,
                usage=record.usage,
            ),
        )

    def _recompute_after_recheck(self, state: PipelineState) -> None:
        """Re-summarize, re-generate candidates, re-analyze after recheck."""
        vp = state.visual_report
        if vp is None:
            return

        summary = self.summarizer.summarize(vp, tuple(state.expert_reports))
        state.accounting.record_call("summarizer", summary.usage)
        state.summarized_report = summary

        cands = self.candidate_generator.generate(summary)
        state.accounting.record_call("candidate_generator", cands.usage)
        state.current_candidates = cands

        disag = self._analyze_disagreement(state)
        state.disagreement = disag

    # ═══════════════════════════════════════════════════
    #  Stage 3: Deliberation
    # ═══════════════════════════════════════════════════

    def _stage_deliberation(self, state: PipelineState) -> None:
        """Run deliberation if DELIBERATION was triggered."""
        should_deliberate = any(
            rd.action == "DELIBERATION" for rd in state.route_decisions
        )
        if not should_deliberate:
            return

        self._run_deliberation(state)

    def _run_deliberation(self, state: PipelineState) -> None:
        """Execute the deliberation. Override in A6 (free debate)."""
        cands = state.current_candidates or state.initial_candidates
        if cands is None:
            return

        result = self.deliberation_manager.deliberate(
            candidates=cands,
            summarized_report=state.summarized_report,
            expert_reports=tuple(state.expert_reports),
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                )
                for r in state.recheck_records
            ),
        )
        # Record deliberation tokens from the result
        for _ in result.rounds:
            state.accounting.record_call(
                "deliberation_hypothesis_a",
                result.usage if result.usage.total_tokens > 0
                else TokenUsage(input_tokens=50, output_tokens=50),
            )
            state.accounting.record_call(
                "deliberation_hypothesis_b",
                result.usage if result.usage.total_tokens > 0
                else TokenUsage(input_tokens=50, output_tokens=50),
            )
            state.accounting.record_call(
                "deliberation_critic",
                result.usage if result.usage.total_tokens > 0
                else TokenUsage(input_tokens=50, output_tokens=50),
            )
        state.deliberation_result = result

    # ═══════════════════════════════════════════════════
    #  Stage 4: Final Judge
    # ═══════════════════════════════════════════════════

    def _stage_final_judge(self, state: PipelineState) -> None:
        """Render the final judgment."""
        cands = state.current_candidates or state.initial_candidates
        if cands is None:
            cands = CandidateSet(candidates=())
        if state.visual_report is None:
            state.visual_report = VisualPerceptionReport(
                content="", usage=TokenUsage()
            )

        final = self.judge.adjudicate(
            image_path=state.image_path,
            visual_report=state.visual_report,
            summarized_report=state.summarized_report
            or SummarizedReport(content="", usage=TokenUsage()),
            candidates=cands,
            expert_reports=tuple(state.expert_reports),
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                )
                for r in state.recheck_records
            ),
            deliberation_result=state.deliberation_result,
        )
        state.accounting.record_call("judge", final.usage)
        state.final_identification = final


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _replace_expert_report(
    reports: list[ExpertReport],
    expert_name: str,
    new_report: ExpertReport,
) -> list[ExpertReport]:
    """Replace the report for a given expert name, or append if not found."""
    for i, r in enumerate(reports):
        if r.expert_name == expert_name:
            reports[i] = new_report
            return reports
    reports.append(new_report)
    return reports
