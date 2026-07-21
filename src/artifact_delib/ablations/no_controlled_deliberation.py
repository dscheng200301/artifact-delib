"""A5: w/o Controlled Deliberation.

When the router decides DELIBERATION, this ablation skips it entirely
and goes directly to Judge. No HypothesisAgent, no CriticAgent.

This is done by overriding _stage_deliberation to be a no-op, so the
pipeline executes all stages EXCEPT deliberation.
"""

from __future__ import annotations

from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.pipeline.state import PipelineState


class AblationNoControlledDeliberation(ArtifactDelibPipeline):
    """A5: Skip deliberation — when router says DELIBERATION, go to Judge directly.

    Overrides _stage_deliberation to be a no-op. HypothesisAgent and
    CriticAgent are never called. Judge still sees all prior context.
    """

    name = "ablation_no_controlled_deliberation"

    def _stage_deliberation(self, state: PipelineState) -> None:
        """No-op: deliberation is skipped entirely.

        If the router triggered DELIBERATION, we intentionally ignore it
        and let the judge work with whatever analysis we have so far.
        """
        # Deliberation is skipped — state.deliberation_result remains None
        pass
