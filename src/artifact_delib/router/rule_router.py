"""Rule-based Dynamic Router — decides which path to take based on candidate disagreement.

Routes:
- FAST: 当前候选已足够稳定，直接进入 Judge
- SHAPE/STYLE/GLYPH/MATERIAL/LOCAL_DETAIL_RECHECK: 定向重审某个专家
- DELIBERATION: 候选仍无法区分，进入受控假设级协商
"""

from __future__ import annotations

from artifact_delib.constants import MAX_RECHECK_ROUNDS
from artifact_delib.schemas import CandidateSet, DisagreementAnalysis, RouteDecision

# Confidence thresholds
HIGH_CONFIDENCE = 0.60
MODERATE_CONFIDENCE = 0.50
LARGE_MARGIN = 0.25
MODERATE_MARGIN = 0.15
LOW_MARGIN = 0.10

# Map from route_hint → route action
HINT_TO_ACTION = {
    "SHAPE": "SHAPE_RECHECK",
    "STYLE": "STYLE_RECHECK",
    "GLYPH": "GLYPH_RECHECK",
    "MATERIAL": "MATERIAL_RECHECK",
    "LOCAL_DETAIL": "LOCAL_DETAIL_RECHECK",
}

# All possible recheck actions
RECHECK_ACTIONS = set(HINT_TO_ACTION.values())

# Route action → human-readable reason
ACTION_REASON = {
    "FAST": "high confidence, sufficient margin",
    "SHAPE_RECHECK": "disagreement in shape features",
    "STYLE_RECHECK": "disagreement in style/period features",
    "GLYPH_RECHECK": "disagreement related to inscription/mark",
    "MATERIAL_RECHECK": "disagreement in material/craft features",
    "LOCAL_DETAIL_RECHECK": "disagreement requires local detail inspection",
    "DELIBERATION": "candidate conflict persists after recheck rounds",
}

# Fallback recheck order when preferred hint is unavailable
_FALLBACK_ORDER = [
    "STYLE_RECHECK",
    "LOCAL_DETAIL_RECHECK",
    "SHAPE_RECHECK",
    "MATERIAL_RECHECK",
    "GLYPH_RECHECK",
]


class RuleRouter:
    """Low-cost deterministic router based on confidence, margin, and disagreement type.

    Consideration order:
    1. If deliberation already done → FAST
    2. If top1 confidence is high AND margin is large → FAST
    3. If max recheck rounds reached AND still uncertain → DELIBERATION
    4. If max recheck rounds reached → FAST (force judgment)
    5. If specific route hint from DisagreementAnalyzer → corresponding RECHECK
       (avoids repeating the same recheck type)
    6. If all recheck types exhausted → DELIBERATION
    7. Default → highest-priority unused recheck
    """

    def __init__(self, max_recheck_rounds: int = MAX_RECHECK_ROUNDS) -> None:
        self.max_recheck_rounds = max(1, max_recheck_rounds)

    def route(
        self,
        disagreement: DisagreementAnalysis | None,
        candidates: CandidateSet,
        recheck_count: int = 0,
        deliberation_count: int = 0,
        completed_rechecks: tuple[str, ...] = (),
    ) -> RouteDecision:
        """Produce a routing decision from current state and history."""
        top1_conf = candidates.top1_confidence
        margin = candidates.margin
        hint = disagreement.route_hint if disagreement else "MULTI_FACTOR"
        done = set(completed_rechecks)

        # Rule 1: Already deliberated → FAST
        if deliberation_count > 0:
            return RouteDecision(
                action="FAST",
                reason=f"deliberation done ({deliberation_count} round(s))",
                recheck_count=recheck_count,
                deliberation_count=deliberation_count,
            )

        # Rule 2: High confidence + large margin → FAST
        if top1_conf >= HIGH_CONFIDENCE and margin >= MODERATE_MARGIN:
            return RouteDecision(
                action="FAST",
                reason=f"high confidence ({top1_conf:.2f}) with margin ({margin:.2f})",
                recheck_count=recheck_count,
            )

        # Rule 3: Max recheck rounds reached
        if recheck_count >= self.max_recheck_rounds:
            # When top1 confidence is below moderate, require larger margin for FAST
            effective_margin_threshold = (
                MODERATE_MARGIN if top1_conf >= MODERATE_CONFIDENCE else LARGE_MARGIN
            )
            if margin < effective_margin_threshold:
                return RouteDecision(
                    action="DELIBERATION",
                    reason=f"uncertain after {recheck_count} recheck(s), margin={margin:.2f}",
                    recheck_count=recheck_count,
                )
            return RouteDecision(
                action="FAST",
                reason=f"max recheck ({recheck_count}) reached, margin={margin:.2f}",
                recheck_count=recheck_count,
            )

        # Rule 4: Specific route hint → corresponding recheck (if not already done)
        preferred_action = HINT_TO_ACTION.get(hint)
        if preferred_action and preferred_action not in done:
            return RouteDecision(
                action=preferred_action,
                reason=ACTION_REASON[preferred_action],
                recheck_count=recheck_count,
            )

        # Rule 5: Preferred action was already done — find another
        next_action = self._find_next_recheck(done, candidates)
        if next_action:
            return RouteDecision(
                action=next_action,
                reason=f"{hint} already done, trying {next_action}",
                recheck_count=recheck_count,
            )

        # Rule 6: All recheck types exhausted
        if recheck_count > 0:
            return RouteDecision(
                action="DELIBERATION",
                reason="all recheck types exhausted, persistent uncertainty",
                recheck_count=recheck_count,
            )

        # Rule 7: Default fallback
        return RouteDecision(
            action="LOCAL_DETAIL_RECHECK",
            reason=f"default fallback (hint={hint})",
            recheck_count=recheck_count,
        )

    def _find_next_recheck(
        self, done: set[str], candidates: CandidateSet
    ) -> str | None:
        """Find the best recheck action not yet performed."""
        available = [a for a in _FALLBACK_ORDER if a not in done]
        if not available:
            return None
        # For very tight margins, prefer style or local detail
        if candidates.margin < LOW_MARGIN:
            for preferred in ["STYLE_RECHECK", "LOCAL_DETAIL_RECHECK"]:
                if preferred in available:
                    return preferred
        return available[0]
