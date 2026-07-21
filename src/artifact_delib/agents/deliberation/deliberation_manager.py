"""Deliberation Manager — orchestrates controlled hypothesis-level deliberation.

Flow:
1. Hypothesis Agent A argues FOR Candidate A
2. Hypothesis Agent B argues FOR Candidate B
3. Critic evaluates whether new discriminative info was produced
4. If yes AND max rounds not reached AND no REVISE/ABSTAIN → repeat
5. Otherwise → stop and return DeliberationResult

Each agent call's usage is tracked in the DeliberationResult for
auditable token accounting.
"""

from __future__ import annotations

from artifact_delib.agents.deliberation.critic_agent import CriticAgent, CriticOutput
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent, HypothesisOutput
from artifact_delib.api.schemas import TokenUsage
from artifact_delib.constants import MAX_DELIBERATION_ROUNDS
from artifact_delib.schemas import (
    CandidateSet,
    DeliberationResult,
    DeliberationRound,
    ExpertReport,
    SummarizedReport,
)


class DeliberationManager:
    """Orchestrate controlled hypothesis-level deliberation.

    Returns DeliberationResult with tracked usage per round:
      Each round = Hypothesis A (1 call) + Hypothesis B (1 call) + Critic (1 call).
    """

    def __init__(
        self,
        hypothesis_agent: HypothesisAgent,
        critic_agent: CriticAgent,
        max_rounds: int = MAX_DELIBERATION_ROUNDS,
    ) -> None:
        self.hypothesis_agent = hypothesis_agent
        self.critic_agent = critic_agent
        self.max_rounds = max(1, max_rounds)

    def deliberate(
        self,
        candidates: CandidateSet,
        summarized_report: SummarizedReport,
        expert_reports: tuple[ExpertReport, ...],
        recheck_reports: tuple[ExpertReport, ...],
    ) -> DeliberationResult:
        """Run controlled deliberation between Top-1 and Top-2 candidates."""
        c1 = candidates.top1
        c2 = candidates.top2
        if c1 is None:
            return DeliberationResult(
                rounds=(), stop_reason="no_candidates",
                summary="没有候选，无需协商。",
            )
        if c2 is None:
            return DeliberationResult(
                rounds=(), stop_reason="insufficient_candidates",
                summary="只有单个候选，无需协商。",
            )

        rounds: list[DeliberationRound] = []
        prior_a_opinions: tuple[str, ...] = ()
        prior_b_opinions: tuple[str, ...] = ()
        prior_feedback: tuple[str, ...] = ()
        total_usage = TokenUsage()
        total_calls = 0

        for round_no in range(1, self.max_rounds + 1):
            # Hypothesis A
            out_a: HypothesisOutput = self.hypothesis_agent.argue(
                candidate_text=c1.text,
                candidate_confidence=c1.confidence,
                opponent_text=c2.text,
                opponent_confidence=c2.confidence,
                summarized_report=summarized_report,
                expert_reports=expert_reports,
                recheck_reports=recheck_reports,
                round_no=round_no,
                prior_opinions=prior_a_opinions,
            )
            total_usage += out_a.usage
            total_calls += 1

            # Hypothesis B
            out_b: HypothesisOutput = self.hypothesis_agent.argue(
                candidate_text=c2.text,
                candidate_confidence=c2.confidence,
                opponent_text=c1.text,
                opponent_confidence=c1.confidence,
                summarized_report=summarized_report,
                expert_reports=expert_reports,
                recheck_reports=recheck_reports,
                round_no=round_no,
                prior_opinions=prior_b_opinions,
            )
            total_usage += out_b.usage
            total_calls += 1

            # Critic
            critic_out: CriticOutput = self.critic_agent.evaluate(
                candidate_a_text=c1.text,
                candidate_b_text=c2.text,
                round_no=round_no,
                opinion_a=out_a.opinion,
                opinion_b=out_b.opinion,
                decision_a=out_a.decision,
                decision_b=out_b.decision,
                prior_feedback=prior_feedback,
            )
            total_usage += critic_out.usage
            total_calls += 1

            dr = DeliberationRound(
                round_no=round_no,
                candidate_a_opinion=out_a.opinion,
                candidate_b_opinion=out_b.opinion,
                candidate_a_decision=out_a.decision,
                candidate_b_decision=out_b.decision,
                critic_feedback=critic_out.feedback,
                hypothesis_a_usage=out_a.usage,
                hypothesis_b_usage=out_b.usage,
                critic_usage=critic_out.usage,
            )
            rounds.append(dr)

            # Stop conditions
            if out_a.decision in ("REVISE", "ABSTAIN") or out_b.decision in ("REVISE", "ABSTAIN"):
                stop = f"{'A' if out_a.decision in ('REVISE','ABSTAIN') else 'B'}_conceded"
                break

            if not critic_out.should_continue:
                stop = "no_new_information"
                break

            # Store for next round
            prior_a_opinions = prior_a_opinions + (out_a.opinion,)
            prior_b_opinions = prior_b_opinions + (out_b.opinion,)
            prior_feedback = prior_feedback + (critic_out.feedback,)
        else:
            stop = "max_rounds"

        summary = self._build_summary(rounds, stop, candidates)
        return DeliberationResult(
            rounds=tuple(rounds),
            stop_reason=stop,
            summary=summary,
            usage=total_usage,
            total_api_calls=total_calls,
        )

    def _build_summary(
        self,
        rounds: list[DeliberationRound],
        stop_reason: str,
        candidates: CandidateSet,
    ) -> str:
        if not rounds:
            return "未进行协商。"
        last = rounds[-1]
        parts = [
            f"经过{len(rounds)}轮受控协商（终止原因：{stop_reason}）：",
            f"候选A（{candidates.top1.text}）最终立场：{last.candidate_a_decision}",
        ]
        if candidates.top2:
            parts.append(
                f"候选B（{candidates.top2.text}）最终立场：{last.candidate_b_decision}"
            )
        return "\n".join(parts)
