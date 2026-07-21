"""Deliberation Manager — orchestrates controlled hypothesis-level deliberation.

Flow:
1. Hypothesis Agent A argues FOR Candidate A
2. Hypothesis Agent B argues FOR Candidate B
3. Critic evaluates whether new discriminative info was produced
4. If yes AND max rounds not reached AND no REVISE/ABSTAIN → repeat
5. Otherwise → stop and return DeliberationResult
"""

from __future__ import annotations

from artifact_delib.agents.deliberation.critic_agent import CriticAgent
from artifact_delib.agents.deliberation.hypothesis_agent import HypothesisAgent
from artifact_delib.constants import MAX_DELIBERATION_ROUNDS
from artifact_delib.schemas import (
    CandidateSet,
    DeliberationResult,
    DeliberationRound,
    ExpertReport,
    SummarizedReport,
)
from artifact_delib.api.schemas import TokenUsage


class DeliberationManager:
    """Orchestrate controlled hypothesis-level deliberation.

    This is Innovation 3 from the paper: not free debate, but structured
    hypothesis comparison with a critic judging discriminative value.
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
        if c1 is None or c2 is None:
            reason = "insufficient_candidates" if c1 else "no_candidates"
            return DeliberationResult(
                rounds=(), stop_reason=reason, summary="只有单个候选，无需协商。",
            )

        rounds: list[DeliberationRound] = []
        prior_a: tuple[str, ...] = ()
        prior_b: tuple[str, ...] = ()
        critic_feedback: tuple[str, ...] = ()
        total_input = 0
        total_output = 0

        for round_no in range(1, self.max_rounds + 1):
            # Round n: Candidate A argues
            opinion_a, decision_a = self.hypothesis_agent.argue(
                candidate_text=c1.text,
                candidate_confidence=c1.confidence,
                opponent_text=c2.text if c2 else "未知",
                opponent_confidence=c2.confidence if c2 else 0.0,
                summarized_report=summarized_report,
                expert_reports=expert_reports,
                recheck_reports=recheck_reports,
                round_no=round_no,
                prior_opinions=prior_a,
            )
            total_input += 0  # tracked inside HypothesisAgent

            # Round n: Candidate B argues
            if c2 is not None:
                opinion_b, decision_b = self.hypothesis_agent.argue(
                    candidate_text=c2.text,
                    candidate_confidence=c2.confidence,
                    opponent_text=c1.text,
                    opponent_confidence=c1.confidence,
                    summarized_report=summarized_report,
                    expert_reports=expert_reports,
                    recheck_reports=recheck_reports,
                    round_no=round_no,
                    prior_opinions=prior_b,
                )
            else:
                opinion_b = "无对方候选"
                decision_b = "ABSTAIN"

            # Critic evaluates
            feedback, should_continue = self.critic_agent.evaluate(
                candidate_a_text=c1.text,
                candidate_b_text=c2.text if c2 else "未知",
                round_no=round_no,
                opinion_a=opinion_a,
                opinion_b=opinion_b,
                decision_a=decision_a,
                decision_b=decision_b,
                prior_feedback=critic_feedback,
            )

            dr = DeliberationRound(
                round_no=round_no,
                candidate_a_opinion=opinion_a,
                candidate_b_opinion=opinion_b,
                candidate_a_decision=decision_a,  # type: ignore[arg-type]
                candidate_b_decision=decision_b,  # type: ignore[arg-type]
                critic_feedback=feedback,
            )
            rounds.append(dr)

            # Stop conditions
            if decision_a in ("REVISE", "ABSTAIN") or decision_b in ("REVISE", "ABSTAIN"):
                stop = f"{'A' if decision_a in ('REVISE','ABSTAIN') else 'B'} {decision_a if decision_a in ('REVISE','ABSTAIN') else decision_b}"
                break

            if not should_continue:
                stop = "no_new_information"
                break

            # Store for next round context
            prior_a = prior_a + (opinion_a,)
            prior_b = prior_b + (opinion_b,)
            critic_feedback = critic_feedback + (feedback,)

        else:
            stop = "max_rounds"

        # Build summary
        summary = self._build_summary(rounds, stop, candidates)
        return DeliberationResult(
            rounds=tuple(rounds),
            stop_reason=stop,
            summary=summary,
        )

    def _build_summary(
        self,
        rounds: list[DeliberationRound],
        stop_reason: str,
        candidates: CandidateSet,
    ) -> str:
        """Build a concise NL summary of the deliberation."""
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
