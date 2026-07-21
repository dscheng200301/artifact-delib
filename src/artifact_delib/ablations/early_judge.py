"""A8: Early Judge instead of Deferred Judge.

The judge makes a preliminary identification right after candidate generation,
BEFORE recheck or deliberation. This preliminary conclusion is then provided
to the recheck and deliberation stages, potentially introducing anchoring bias.

The full pipeline defers the judge until after all recheck/deliberation.
"""
from __future__ import annotations
from pathlib import Path
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import CandidateSet, PipelineResult


class AblationEarlyJudge(ArtifactDelibPipeline):
    """A8: Judge runs early (before recheck/deliberation).

    The early judge result IS the final result. Recheck and deliberation
    are not run. This tests whether deferring the judge reduces anchoring
    bias and improves outcomes.
    """
    name = "ablation_early_judge"

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        total_calls = 0

        # Standard steps
        vp = self.visual_perception.analyze(image_path)
        total_calls += 1

        reports = []
        for expert in [
            self.shape_expert, self.style_expert, self.glyph_expert,
            self.material_expert, self.local_detail_expert,
        ]:
            reports.append(expert.analyze(image_path))
            total_calls += 1
        et = tuple(reports)

        summary = self.summarizer.summarize(vp, et)
        total_calls += 1

        candidates = self.candidate_generator.generate(summary)
        total_calls += 1

        # Judge runs EARLY — before recheck or deliberation
        early_final = self.judge.adjudicate(
            image_path=image_path,
            visual_report=vp,
            summarized_report=summary,
            candidates=candidates,
            expert_reports=et,
        )
        total_calls += 1

        return PipelineResult(
            sample_id=sample_id,
            final_identification=early_final,
            visual_perception_report=vp,
            expert_reports=et,
            summarized_report=summary,
            initial_candidates=candidates,
            disagreement_analysis=None,
            total_usage=early_final.usage,
            total_api_calls=total_calls,
            status="COMPLETED",
        )
