"""A8: Early Judge instead of Deferred Judge.

Tests whether early judge participation introduces anchoring bias.

Full ArtifactDelib: Judge is the LAST module (deferred).
A8: Judge runs AFTER candidate generation and its preliminary opinion
    is visible to subsequent recheck/deliberation stages AND the final judge.

Overrides _stage_initial_analysis to produce a preliminary judgment.
Overrides _stage_final_judge to pass the early judgment as context to the judge.
"""

from __future__ import annotations

from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.pipeline.state import PipelineState
from artifact_delib.api.schemas import TokenUsage
from artifact_delib.schemas import (
    CandidateSet,
    ExpertReport,
    FinalIdentification,
    SummarizedReport,
    VisualPerceptionReport,
)


class AblationEarlyJudge(ArtifactDelibPipeline):
    """A8: Judge runs early (before recheck/deliberation).

    The preliminary judgment is injected into the pipeline state, making
    it visible to subsequent recheck and deliberation stages — testing
    whether anchoring bias affects final outcomes. The final judge sees
    the early judgment as additional context.

    Research question: Does deferring the judge reduce anchoring bias?
    Compare: Full pipeline (deferred judge) vs A8 (early judge visible).
    """

    name = "ablation_early_judge"

    def _stage_initial_analysis(self, state: PipelineState) -> None:
        """Run initial analysis AND early judge."""
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

    def _stage_final_judge(self, state: PipelineState) -> None:
        """Final judge sees the early judgment as context (anchoring bias test).

        The early judgment is prepended to the summarized report, so the
        final judge sees it alongside all other evidence. This simulates
        a scenario where the judge forms an early opinion that is then
        visible during final deliberation.

        If full pipeline accuracy > A8 accuracy, it suggests that deferring
        the judge (so it can't anchor subsequent reasoning) is beneficial.
        """
        if state.preliminary_judgment is None:
            return

        cands = state.current_candidates or state.initial_candidates
        if cands is None:
            cands = CandidateSet(candidates=())
        vp = state.visual_report
        if vp is None:
            vp = VisualPerceptionReport(content="", usage=TokenUsage())
        summary = state.summarized_report
        if summary is None:
            summary = SummarizedReport(content="", usage=TokenUsage())

        # Create a version of the summarized report that includes the early
        # judgment — this is the anchoring bias manipulation
        anchored_summary = SummarizedReport(
            content=(
                f"【早期法官初步判断】{state.preliminary_judgment.content}\n\n"
                f"【专家综合分析】\n{summary.content}"
            ),
            usage=summary.usage,
        )

        final = self.judge.adjudicate(
            image_path=state.image_path,
            visual_report=vp,
            summarized_report=anchored_summary,
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
        state.accounting.record_call("final_judge", final.usage)
        state.final_identification = final