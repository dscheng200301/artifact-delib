"""Critic Agent — evaluates controlled deliberation for new discriminative information.

The critic judges whether the current round of deliberation has produced
any genuinely new evidence or reasoning that helps distinguish the candidates.

Returns CriticOutput dataclass with feedback, should_continue flag, and usage
for accurate token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.api.schemas import TokenUsage


@dataclass(frozen=True)
class CriticOutput:
    """Output from a CriticAgent — feedback + continue flag + usage."""
    feedback: str
    should_continue: bool
    usage: TokenUsage = field(default_factory=TokenUsage)


class CriticAgent(ArtifactAgent):
    """Evaluates deliberation rounds and decides whether to continue.

    The critic does NOT decide which candidate is correct.
    It only judges whether new discriminative information was produced.
    """

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "critic", model_name)

    def evaluate(
        self,
        candidate_a_text: str,
        candidate_b_text: str,
        round_no: int,
        opinion_a: str,
        opinion_b: str,
        decision_a: str,
        decision_b: str,
        prior_feedback: tuple[str, ...] = (),
    ) -> CriticOutput:
        """Evaluate one deliberation round.

        Returns:
            CriticOutput with feedback, should_continue, and usage.
        """
        system = (
            "你是古代文物鉴定协商过程的评估专家（Critic）。\n\n"
            "你的任务不是判断哪个候选正确，而是评估当前这轮协商：\n"
            "1. 双方是否提出了新的、有区分力的信息？\n"
            "2. 这些信息是否基于已有的专家分析而非猜测？\n"
            "3. 是否有一方已经修正立场（REVISE）或放弃判断（ABSTAIN）？\n\n"
            "如果出现以下任一情况，应停止协商：\n"
            "- 某一方 REVISE 或 ABSTAIN\n"
            "- 双方没有提出任何新的区分信息\n"
            "- 已经达到最大轮数\n\n"
            "输出格式：\n"
            "先输出一段简短评估（50-100字），"
            "然后在最后一行单独输出'继续：是'或'继续：否'。"
        )
        user = (
            f"第{round_no}轮协商评估：\n\n"
            f"候选A（{candidate_a_text}）：\n{opinion_a}\n"
            f"→ 立场：{decision_a}\n\n"
            f"候选B（{candidate_b_text}）：\n{opinion_b}\n"
            f"→ 立场：{decision_b}\n\n"
            f"请评估这一轮是否产生了新的有效区分信息，是否应该继续协商。"
        )
        if prior_feedback:
            user += f"\n\n以往评估：\n" + "\n".join(
                f"  第{i+1}轮：{fb[:80]}..." for i, fb in enumerate(prior_feedback)
            )

        content, usage = self._call(system, user, max_output_tokens=256)
        should_continue = self._extract_continue(content)
        return CriticOutput(
            feedback=content.strip(),
            should_continue=should_continue,
            usage=usage,
        )

    def _extract_continue(self, content: str) -> bool:
        """Extract whether to continue from the response."""
        import re
        match = re.search(r'继续[：:]\s*([是是否否])', content)
        if match:
            return match.group(1) == "是"
        # Fallback: check last line
        last = content.strip().split("\n")[-1].strip()
        if "否" in last or "停止" in last:
            return False
        if "是" in last or "继续" in last:
            return True
        return False
