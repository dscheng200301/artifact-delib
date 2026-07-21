"""A3: w/o Dynamic Routing — fixed full path for every sample.

All samples execute the same path:
  VP -> 5 Experts -> Summarizer -> Candidates -> Recheck -> Deliberation -> Judge

No FAST path, no targeted recheck, no conditional deliberation.
Same as FixedFullBaseline but using pipeline classes.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.constants import MAX_DELIBERATION_ROUNDS, MAX_RECHECK_ROUNDS
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import PipelineResult


class AblationNoDynamicRouting(ArtifactDelibPipeline):
    """A3: Fixed full path — VP -> 5 experts -> recheck -> deliberation -> judge.

    No conditional routing. Every sample runs the full pipeline including
    all 5 rechecks and deliberation.
    """

    name = "ablation_no_dynamic_routing"

    def run(
        self,
        image_path: Path,
        sample_id: str = "unknown",
    ) -> PipelineResult:
        """Run the full fixed pipeline without adaptive routing."""
        from artifact_delib.schemas import ExpertReport, RouteDecision

        total_calls = 0

        # VP
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        # 5 Experts
        reports = []
        for expert in [
            self.shape_expert, self.style_expert, self.glyph_expert,
            self.material_expert, self.local_detail_expert,
        ]:
            reports.append(expert.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        # Summarize
        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        # Candidates
        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        # Disagreement
        disagreement = self.disagreement_analyzer.analyze(candidates, summary)
        total_calls += 1

        # Run all 5 rechecks (no routing)
        recheck_records = []
        for action in (
            "SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
            "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK",
        ):
            route = RouteDecision(action=action, reason="fixed", recheck_count=0)
            record = self.recheck_coordinator.execute(
                image_path, route, candidates, disagreement, et,
                recheck_history=tuple(recheck_records),
                round_no=len(recheck_records) + 1,
            )
            recheck_records.append(record)
            total_calls += 1

        # Re-summarize
        expert_reports_upd = []
        for exp_name in ["shape", "style", "glyph", "material", "local_detail"]:
            last = next(
                (r for r in reversed(recheck_records) if r.expert_name in exp_name),
                None,
            )
            if last:
                expert_reports_upd.append(ExpertReport(
                    expert_name=exp_name, content=last.new_content, usage=last.usage,
                ))
            else:
                for r in et:
                    if r.expert_name == exp_name:
                        expert_reports_upd.append(r)
                        break
        et_upd = tuple(expert_reports_upd)
        summary = self.summarizer.summarize(vp, et_upd)
        total_calls += 1

        # Deliberation (always run)
        deliberation_result = self.deliberation_manager.deliberate(
            candidates, summary, et_upd,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
        )
        for dr in deliberation_result.rounds:
            total_calls += 3

        # Judge
        final = self.judge.adjudicate(
            image_path, vp, summary, candidates, et_upd,
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            deliberation_result=deliberation_result,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=final,
            visual_perception_report=vp,
            expert_reports=et_upd,
            summarized_report=summary,
            initial_candidates=candidates,
            disagreement_analysis=disagreement,
            route_decisions=tuple(
                RouteDecision(action=a, reason="fixed", recheck_count=0)
                for a in ("SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
                          "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK")
            ),
            recheck_reports=tuple(
                ExpertReport(expert_name=r.expert_name, content=r.new_content, usage=r.usage)
                for r in recheck_records
            ),
            recheck_records=tuple(recheck_records),
            deliberation_result=deliberation_result,
            total_usage=reports[0].usage if reports else final.usage,
            total_api_calls=total_calls,
            status="COMPLETED",
        )
