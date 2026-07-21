"""A8: Early Judge instead of Deferred Judge.

Tests whether early judge participation introduces anchoring bias.

Full ArtifactDelib: Judge is the LAST module (deferred).
A8: Judge runs AFTER candidate generation and its preliminary opinion
    is shown to recheck/deliberation stages.

Overrides _stage_initial_analysis to produce a preliminary judgment
that is passed to subsequent rechecks and deliberation.
"""

from __future__ import annotations

from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.pipeline.state import PipelineState
from artifact_delib.api.schemas import TokenUsage
from artifact_delib.schemas import CandidateSet, SummarizedReport, VisualPerceptionReport


class AblationEarlyJudge(ArtifactDelibPipeline):
    """A8: Judge runs early (before recheck/deliberation).

    The preliminary judgment is injected into the pipeline state, making
    it visible to subsequent recheck and deliberation stages — testing
    whether anchoring bias affects final outcomes.

    The pipeline still runs the full routing loop and final judge.
    """

    name = "ablation_early_judge"

    def _stage_initial_analysis(self, state: PipelineState) -> None:
        """Run initial analysis AND early judge."""
        # Run standard initial analysis first (VP, experts, summarizer, candidates, disagreement)
        super()._stage_initial_analysis(state)

        # Early judge: runs BEFORE routing/recheck/deliberation
        cands = state.current_candidates or state.initial_candidates
        if cands is None:
            cands = CandidateSet(candidates=())
        vp = state.visual_report
        if vp is None:
            vp = VisualPerceptionReport(content="", usage=TokenUsage())
        summary = state.summarized_report
        if summary is None:
            summary = SummarizedReport(content="", usage=TokenUsage())

        early_judgment = self.judge.adjudicate(
            image_path=state.image_path,
            visual_report=vp,
            summarized_report=summary,
            candidates=cands,
            expert_reports=tuple(state.expert_reports),
        )
        state.accounting.record_call("early_judge", early_judgment.usage)
        state.preliminary_judgment = early_judgment

        # NOTE: The preliminary judgment is available in PipelineState.
        # Downstream stages (recheck prompts, deliberation) can reference it,
        # which may introduce anchoring bias. The full pipeline defers the
        # judge until ALL recheck/deliberation is done, preventing this.

    def _stage_final_judge(self, state: PipelineState) -> None:
        """Final judge sees the early judgment (anchoring bias test).

        In the no-ablation pipeline, the judge is deferred and sees only
        the final state. Here, we don't run a second judge call — the
        early judgment IS the final result. This lets us compare:
          Full pipeline result vs. Early judge result
        and measure how much anchoring matters.
        """
        # The early judgment is the final result — this tests whether
        # deferring the judge (full pipeline) improves over acting early.
        state.final_identification = state.preliminary_judgment
