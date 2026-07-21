"""A2: w/o Candidate-Disagreement Analysis.

Truly replaces semantic disagreement analysis with margin-only routing.
Does NOT run the full pipeline and then modify results — the routing
decisions are made by margin-only logic and the real pipeline follows
those decisions.
"""

from __future__ import annotations

from artifact_delib.constants import MAX_RECHECK_ROUNDS
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.pipeline.state import PipelineState
from artifact_delib.router.rule_router import (
    HIGH_CONFIDENCE,
    LARGE_MARGIN,
    LOW_MARGIN,
    MODERATE_CONFIDENCE,
    MODERATE_MARGIN,
    _FALLBACK_ORDER,
)
from artifact_delib.schemas import CandidateSet, RouteDecision


class AblationNoDisagreementAnalysis(ArtifactDelibPipeline):
    """A2: Margin-only routing — no semantic disagreement analysis.

    Overrides _make_routing_decision to use only confidence/margin.
    The DisagreementAnalyzer still runs (for fair comparison) but its
    route_hint is intentionally ignored.
    """

    name = "ablation_no_disagreement_analysis"

    def _make_routing_decision(self, state: PipelineState) -> RouteDecision:
        """Margin-only routing: no semantic disagreement type."""
        candidates = (
            state.current_candidates
            or state.initial_candidates
            or CandidateSet(candidates=())
        )
        top1_conf = candidates.top1_confidence
        margin = candidates.margin
        done = state.completed_rechecks

        # Already deliberated → FAST
        if state.deliberation_count > 0:
            return RouteDecision(
                action="FAST",
                reason="margin-only: deliberation already done",
                recheck_count=state.recheck_count,
            )

        # High confidence + margin → FAST
        if top1_conf >= HIGH_CONFIDENCE and margin >= MODERATE_MARGIN:
            return RouteDecision(
                action="FAST",
                reason=f"margin-only: high conf ({top1_conf:.2f}) margin ({margin:.2f})",
                recheck_count=state.recheck_count,
            )

        # Max recheck reached
        if state.recheck_count >= self.max_recheck_rounds:
            effective_threshold = (
                MODERATE_MARGIN if top1_conf >= MODERATE_CONFIDENCE
                else LARGE_MARGIN
            )
            if margin < effective_threshold:
                return RouteDecision(
                    action="DELIBERATION",
                    reason=f"margin-only: uncertain after {state.recheck_count} rechecks",
                    recheck_count=state.recheck_count,
                )
            return RouteDecision(
                action="FAST",
                reason=f"margin-only: max recheck reached, margin acceptable",
                recheck_count=state.recheck_count,
            )

        # Pick next unused recheck (no semantic routing)
        available = [a for a in _FALLBACK_ORDER if a not in done]
        if not available:
            return RouteDecision(
                action="DELIBERATION",
                reason="margin-only: all recheck types exhausted",
                recheck_count=state.recheck_count,
            )

        # For tight margins, prefer STYLE or LOCAL_DETAIL first
        if margin < LOW_MARGIN:
            for preferred in ["STYLE_RECHECK", "LOCAL_DETAIL_RECHECK"]:
                if preferred in available:
                    return RouteDecision(
                        action=preferred,
                        reason=f"margin-only: tight margin ({margin:.2f}), trying {preferred}",
                        recheck_count=state.recheck_count,
                    )

        action = available[0]
        return RouteDecision(
            action=action,
            reason=f"margin-only: fallback recheck ({action}), margin={margin:.2f}",
            recheck_count=state.recheck_count,
        )
