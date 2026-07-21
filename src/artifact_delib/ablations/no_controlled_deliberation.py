"""A5: w/o Controlled Deliberation.

When the router decides DELIBERATION, go straight to Judge instead.
This tests whether hard samples benefit from the deliberation process.

Tracking: we record the deliberation_triggered flag even though we skip it.
"""
from __future__ import annotations
from pathlib import Path
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import PipelineResult


class AblationNoControlledDeliberation(ArtifactDelibPipeline):
    """A5: Skip deliberation — when router says DELIBERATION, go directly to Judge."""
    name = "ablation_no_controlled_deliberation"

    def run(self, image_path: Path, sample_id: str = "unknown") -> PipelineResult:
        result = super().run(image_path, sample_id)
        if result.deliberation_result and result.deliberation_result.rounds:
            # Nullify the deliberation result
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
