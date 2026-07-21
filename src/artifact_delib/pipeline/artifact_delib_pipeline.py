"""ArtifactDelib pipeline — Phase 4: with deliberation loop."""

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
    """ArtifactDelib pipeline with full routing + targeted recheck (Phase 3).

    Flow:
    Image → VP → 5 Experts → Summarizer → Top-K → Disagreement →
      Router → [FAST | Recheck (with version tracking) → Loop | Deliberation] → Judge
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

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run full pipeline with targeted recheck and version tracking."""
        total_input = 0
        total_output = 0
        total_calls = 0

        # ───────── Step 1: Visual Perception ─────────
        visual_report = self.visual_perception.analyze(image_path)
        total_input += visual_report.usage.input_tokens
        total_output += visual_report.usage.output_tokens
        total_calls += 1

        # ───────── Step 2: Multi-Expert Analysis ─────────
        expert_reports: list[ExpertReport] = []
        for expert in [
            self.shape_expert,
            self.style_expert,
            self.glyph_expert,
            self.material_expert,
            self.local_detail_expert,
        ]:
            report = expert.analyze(image_path)
            expert_reports.append(report)
            total_input += report.usage.input_tokens
            total_output += report.usage.output_tokens
            total_calls += 1

        # ───────── Step 3: Report Summarizer ─────────
        summarized = self.summarizer.summarize(visual_report, tuple(expert_reports))
        total_input += summarized.usage.input_tokens
        total_output += summarized.usage.output_tokens
        total_calls += 1

        # ───────── Step 4: Candidate Generator ─────────
        candidates = self.candidate_generator.generate(summarized)
        total_input += candidates.usage.input_tokens
        total_output += candidates.usage.output_tokens
        total_calls += 1

        # ───────── Step 5: Disagreement Analysis + Routing Loop ─────────
        disagreement = self.disagreement_analyzer.analyze(candidates, summarized)
        total_input += disagreement.usage.input_tokens
        total_output += disagreement.usage.output_tokens
        total_calls += 1

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
                    summarized_report=summarized,
                    expert_reports=tuple(expert_reports),
                    recheck_reports=tuple(
                        ExpertReport(
                            expert_name=r.expert_name,
                            content=r.new_content,
                            usage=r.usage,
                        )
                        for r in recheck_records
                    ),
                )
                # Add deliberation API calls to total (each round = 3 calls: A+B+critic)
                for dr in deliberation_result.rounds:
                    total_calls += 3
                break

            # ── Targeted Recheck via coordinator ──
            if route.action in _ACTION_TO_EXPERT_NAME:
                record = self.recheck_coordinator.execute(
                    image_path=image_path,
                    route=route,
                    candidates=candidates,
                    disagreement=disagreement,
                    current_reports=tuple(expert_reports),
                    recheck_history=tuple(recheck_records),
                    round_no=recheck_count + 1,
                )
                total_input += record.usage.input_tokens
                total_output += record.usage.output_tokens
                total_calls += 1

                recheck_records.append(record)
                recheck_count += 1
                completed_rechecks.append(route.action)

                # Update expert_reports (replace old with new)
                expert_reports = _replace_expert_report(
                    expert_reports,
                    _ACTION_TO_EXPERT_NAME[route.action],
                    ExpertReport(
                        expert_name=_ACTION_TO_EXPERT_NAME[route.action],
                        content=record.new_content,
                        usage=record.usage,
                    ),
                )

                # Re-summarize
                summarized = self.summarizer.summarize(
                    visual_report, tuple(expert_reports)
                )
                total_input += summarized.usage.input_tokens
                total_output += summarized.usage.output_tokens
                total_calls += 1

                # Re-generate candidates
                candidates = self.candidate_generator.generate(summarized)
                total_input += candidates.usage.input_tokens
                total_output += candidates.usage.output_tokens
                total_calls += 1

                # Re-analyze disagreement
                disagreement = self.disagreement_analyzer.analyze(candidates, summarized)
                total_input += disagreement.usage.input_tokens
                total_output += disagreement.usage.output_tokens
                total_calls += 1

                continue

            break

        # ───────── Final Step: Deferred Artifact Judge ─────────
        final = self.judge.adjudicate(
            image_path=image_path,
            visual_report=visual_report,
            summarized_report=summarized,
            candidates=candidates,
            expert_reports=tuple(expert_reports),
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                )
                for r in recheck_records
            ),
            deliberation_result=deliberation_result,
        )
        total_input += final.usage.input_tokens
        total_output += final.usage.output_tokens
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=visual_report,
            expert_reports=tuple(expert_reports),
            summarized_report=summarized,
            initial_candidates=initial_candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(route_decisions),
            recheck_reports=tuple(
                ExpertReport(
                    expert_name=r.expert_name,
                    content=r.new_content,
                    usage=r.usage,
                )
                for r in recheck_records
            ),
            updated_candidates=(
                candidates if len(route_decisions) > 1 else None
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
            ),
            total_api_calls=total_calls,
            status="COMPLETED",
        )


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
