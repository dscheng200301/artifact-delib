"""Oracle Route Builder — compute optimal route for each training sample.

For each sample with a known gold label, this module:
1. Runs the full pipeline and captures intermediate features (candidate confidences, etc.)
2. For each possible route action (FAST, RECHECK variants, DELIBERATION),
   simulates what would happen and checks if the final answer would be correct
3. Records the lowest-cost route that leads to a correct answer

The output is an OracleRouteDataset suitable for training a Learned Router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_delib.schemas import RouteDecision


@dataclass(frozen=True)
class RouteFeatures:
    """Features available to the router at decision time."""

    top1_confidence: float
    top2_confidence: float
    margin: float
    disagreement_type: str  # SHAPE, STYLE, GLYPH, MATERIAL, LOCAL_DETAIL, MULTI_FACTOR
    n_candidates: int = 3


@dataclass(frozen=True)
class RouteOutcome:
    """Outcome of trying a specific route for a sample."""

    route_action: str
    is_correct: bool
    api_calls: int = 0
    total_tokens: int = 0
    final_type_correct: bool = False
    final_period_correct: bool = False
    joint_correct: bool = False


@dataclass(frozen=True)
class OracleRecord:
    """One record in the oracle dataset: features → optimal route."""

    sample_id: str
    features: RouteFeatures
    oracle_route: str  # The lowest-cost route that yields correct answer
    outcomes: tuple[RouteOutcome, ...]  # All tried routes for reference


class OracleRouteBuilder:
    """Offline enumeration to find oracle routes for training data.

    This module REQUIRES gold labels — it is only used during training/development,
    never during inference or testing.

    ROUTE_ACTIONS enumerates the 7 possible decisions the Learned Router must learn.
    """

    ROUTE_ACTIONS = (
        "FAST",
        "SHAPE_RECHECK",
        "STYLE_RECHECK",
        "GLYPH_RECHECK",
        "MATERIAL_RECHECK",
        "LOCAL_DETAIL_RECHECK",
        "DELIBERATION",
    )

    def __init__(
        self,
        parser,
        metrics,
        max_recheck_rounds: int = 2,
    ) -> None:
        self.parser = parser
        self.metrics = metrics
        self.max_recheck_rounds = max_recheck_rounds

    def extract_features(self, pipeline_result) -> RouteFeatures:
        """Extract router-relevant features from pipeline state."""
        candidates = pipeline_result.initial_candidates
        top1 = candidates.top1
        top2 = candidates.top2
        hint = "MULTI_FACTOR"
        if pipeline_result.disagreement_analysis:
            hint = pipeline_result.disagreement_analysis.route_hint

        return RouteFeatures(
            top1_confidence=top1.confidence if top1 else 0.0,
            top2_confidence=top2.confidence if top2 else 0.0,
            margin=candidates.margin,
            disagreement_type=hint,
            n_candidates=len(candidates.candidates),
        )

    def score_route(
        self,
        route_action: str,
        final_text: str,
        gold_category: str | None,
        gold_type: str | None,
        gold_period: str | None,
        api_calls: int,
        tokens: int,
    ) -> RouteOutcome:
        """Score a route by parsing the final answer and checking correctness."""
        parsed = self.parser.parse(final_text)
        is_type_correct = (
            parsed.fine_grained_type == gold_type
            if parsed.fine_grained_type and gold_type
            else False
        )
        is_period_correct = (
            parsed.period == gold_period
            if parsed.period and gold_period
            else False
        )
        is_joint = is_type_correct and is_period_correct

        return RouteOutcome(
            route_action=route_action,
            is_correct=is_type_correct,  # Primary: type accuracy
            api_calls=api_calls,
            total_tokens=tokens,
            final_type_correct=is_type_correct,
            final_period_correct=is_period_correct,
            joint_correct=is_joint,
        )

    def select_oracle_route(
        self, outcomes: list[RouteOutcome]
    ) -> tuple[str, RouteOutcome | None]:
        """Select the oracle route: lowest-cost correct route.

        Tie-breaking:
        1. Prefer correct over incorrect (always)
        2. Fewer API calls
        3. Fewer tokens
        4. Prefer FAST (simplest) if tied
        """
        correct = [o for o in outcomes if o.is_correct]
        if not correct:
            # No route correct — pick the cheapest regardless
            cheapest = min(outcomes, key=lambda o: (o.api_calls, o.total_tokens))
            return cheapest.route_action, cheapest

        # Among correct routes, pick cheapest (API calls, then tokens)
        best = min(correct, key=lambda o: (
            o.api_calls, o.total_tokens,
            0 if o.route_action == "FAST" else 1,  # Prefer FAST if tied
        ))
        return best.route_action, best

    def compute_relative_cost(
        self, outcome: RouteOutcome, oracle_outcome: RouteOutcome
    ) -> float:
        """Compute cost ratio relative to oracle route."""
        if oracle_outcome.api_calls == 0:
            return 1.0
        return outcome.api_calls / oracle_outcome.api_calls


# ─────────── Export helpers ───────────

def features_to_tensor(features: RouteFeatures) -> list[float]:
    """Convert route features to a fixed-size numeric vector for MLP input."""
    disagreement_idx = {
        "SHAPE": 0, "STYLE": 1, "GLYPH": 2,
        "MATERIAL": 3, "LOCAL_DETAIL": 4, "MULTI_FACTOR": 5,
    }
    return [
        features.top1_confidence,
        features.top2_confidence,
        features.margin,
        float(disagreement_idx.get(features.disagreement_type, 5)),
        float(features.n_candidates),
        # Additional derived features
        features.top1_confidence - features.top2_confidence,
        features.top1_confidence * features.top2_confidence,
    ]


def route_to_label(route_action: str) -> int:
    """Convert route action to integer class label."""
    labels = {
        "FAST": 0, "SHAPE_RECHECK": 1, "STYLE_RECHECK": 2,
        "GLYPH_RECHECK": 3, "MATERIAL_RECHECK": 4,
        "LOCAL_DETAIL_RECHECK": 5, "DELIBERATION": 6,
    }
    return labels.get(route_action, 0)


def label_to_route(label: int) -> str:
    """Convert integer class label back to route action."""
    routes = [
        "FAST", "SHAPE_RECHECK", "STYLE_RECHECK", "GLYPH_RECHECK",
        "MATERIAL_RECHECK", "LOCAL_DETAIL_RECHECK", "DELIBERATION",
    ]
    return routes[label] if 0 <= label < len(routes) else "FAST"

# N_FEATURES must match features_to_tensor output length
N_FEATURES = 7
N_CLASSES = 7
