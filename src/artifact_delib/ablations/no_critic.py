"""A7: w/o Critic — deliberation without CriticAgent.

Hypothesis A argues for Top-1, Hypothesis B argues for Top-2.
No CriticAgent evaluates their arguments. Judge receives both
arguments directly and makes the final decision.

This isolates the CriticAgent's contribution to the deliberation quality.
"""
from __future__ import annotations
from pathlib import Path
from artifact_delib.api.base import ModelClient
from artifact_delib.constants import DEFAULT_TOP_K, MAX_DELIBERATION_ROUNDS, MAX_RECHECK_ROUNDS
from artifact_delib.agents.deliberation.deliberation_manager import DeliberationManager
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.schemas import DeliberationResult, DeliberationRound, ExpertReport, PipelineResult


class _NoCriticDeliberationManager:
    """Deliberation manager without CriticAgent.

    Hypothesis A and B argue, then we stop (no Critic feedback).
    The judge sees both argument sides directly.
    """

    def __init__(self, hypothesis_agent: HypothesisAgent, max_rounds: int = 2) -> None:
        self.hypothesis_agent = hypothesis_agent
        self.max_rounds = max_rounds

    def deliberate(
        self,
        candidates,
        summarized_report,
        expert_reports,
        recheck_reports=(),
    ) -> DeliberationResult:
        if len(candidates.candidates) < 2:
            return DeliberationResult(
                rounds=(),
                stop_reason="insufficient_candidates",
                summary="",
            )

        top1 = candidates.candidates[0]
        top2 = candidates.candidates[1]

        rounds = []
        for rnd in range(1, self.max_rounds + 1):
            opinion_a, decision_a = self.hypothesis_agent.argue(
                candidate_text=top1.text,
                candidate_confidence=top1.confidence,
                opponent_text=top2.text,
                opponent_confidence=top2.confidence,
                summarized_report=summarized_report,
                expert_reports=expert_reports,
                recheck_reports=recheck_reports,
                round_no=rnd,
            )
            opinion_b, decision_b = self.hypothesis_agent.argue(
                candidate_text=top2.text,
                candidate_confidence=top2.confidence,
                opponent_text=top1.text,
                opponent_confidence=top1.confidence,
                summarized_report=summarized_report,
                expert_reports=expert_reports,
                recheck_reports=recheck_reports,
                round_no=rnd,
            )

            rounds.append(DeliberationRound(
                round_no=rnd,
                candidate_a_opinion=opinion_a,
                candidate_b_opinion=opinion_b,
                candidate_a_decision=decision_a,
                candidate_b_decision=decision_b,
                critic_feedback="(skipped - no critic)",
            ))

            if decision_a == decision_b and decision_a == "MAINTAIN":
                break

        return DeliberationResult(
            rounds=tuple(rounds),
            stop_reason="no_critic_max_rounds" if len(rounds) >= self.max_rounds else "no_critic_consensus",
            summary=f"No-critic deliberation: {len(rounds)} round(s), no CriticAgent used.",
        )


class AblationNoCritic(ArtifactDelibPipeline):
    """A7: Deliberation without CriticAgent.

    Replaces the deliberation manager with one that skips CriticAgent.
    The rest of the pipeline is identical.
    """
    name = "ablation_no_critic"

    def __init__(
        self,
        client: ModelClient,
        model_name: str = "default",
        top_k: int = DEFAULT_TOP_K,
        max_recheck_rounds: int = MAX_RECHECK_ROUNDS,
    ) -> None:
        super().__init__(client, model_name, top_k, max_recheck_rounds)
        # Replace deliberation manager with no-critic version
        self.deliberation_manager = _NoCriticDeliberationManager(
            hypothesis_agent=HypothesisAgent(client, model_name),
            max_rounds=MAX_DELIBERATION_ROUNDS,
        )
