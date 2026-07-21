"""Hypothesis Agent — represents one candidate position in controlled deliberation.

Not a free-form debater. Each hypothesis agent argues FOR one candidate,
based strictly on the existing expert reports and visual analysis.

Returns HypothesisOutput dataclass with opinion, decision, and usage for
accurate token accounting in deliberation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.api.schemas import TokenUsage
from artifact_delib.schemas import ExpertReport, SummarizedReport


@dataclass(frozen=True)
class HypothesisOutput:
    """Output from a HypothesisAgent — opinion + decision + usage."""
    opinion: str
    decision: str  # MAINTAIN, REVISE, or ABSTAIN
    usage: TokenUsage = field(default_factory=TokenUsage)


class HypothesisAgent(ArtifactAgent):
    """Represents one candidate hypothesis in deliberation.

    Argues FOR the assigned candidate based on existing evidence,
    then states: MAINTAIN, REVISE, or ABSTAIN.
    """

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "hypothesis_agent", model_name)

    def argue(
        self,
        candidate_text: str,
        candidate_confidence: float,
        opponent_text: str,
        opponent_confidence: float,
        summarized_report: SummarizedReport,
        expert_reports: tuple[ExpertReport, ...],
        recheck_reports: tuple[ExpertReport, ...],
        round_no: int,
        prior_opinions: tuple[str, ...] = (),
    ) -> HypothesisOutput:
        """Argue for the assigned candidate and return HypothesisOutput.

        Returns:
            HypothesisOutput with opinion text, decision tag, and usage.
        """
        system = (
            f"你是古代文物鉴定专家。当前需要进行受控假设级协商。\n\n"
            f"你被分配支持以下候选：{candidate_text}（置信度{candidate_confidence:.2f}）\n"
            f"对方候选：{opponent_text}（置信度{opponent_confidence:.2f}）\n\n"
            f"这是第{round_no}轮协商。\n\n"
            "你的任务：\n"
            "1. 基于已有的专家分析和视觉证据，说明为什么你的候选更合理\n"
            "2. 如果对方的证据更有说服力，可以修正立场\n"
            "3. 最后从以下选择一种立场：MAINTAIN（维持）、REVISE（修正为对方候选）、ABSTAIN（无法判断）\n\n"
            "输出格式：\n"
            "先输出一段简短专业意见（100-200字），"
            "然后在最后一行单独输出'立场：MAINTAIN'、'立场：REVISE'或'立场：ABSTAIN'。"
        )

        evidence = self._build_evidence(
            summarized_report, expert_reports, recheck_reports, prior_opinions
        )
        content, usage = self._call(system, evidence, max_output_tokens=512)
        decision = self._extract_decision(content)
        return HypothesisOutput(
            opinion=content.strip(),
            decision=decision,
            usage=usage,
        )

    def _build_evidence(
        self,
        summarized: SummarizedReport,
        expert_reports: tuple[ExpertReport, ...],
        recheck_reports: tuple[ExpertReport, ...],
        prior_opinions: tuple[str, ...],
    ) -> str:
        parts = [f"综合视觉分析报告：\n{summarized.content}\n"]
        parts.append("专家分析报告：")
        for r in expert_reports:
            parts.append(f"  【{r.expert_name}】{r.content}")
        if recheck_reports:
            parts.append("\n定向重审报告：")
            for r in recheck_reports:
                parts.append(f"  【{r.expert_name}重审】{r.content}")
        if prior_opinions:
            parts.append("\n以往轮次意见：")
            for i, op in enumerate(prior_opinions, 1):
                parts.append(f"  第{i}轮：{op[:100]}...")
        return "\n".join(parts)

    def _extract_decision(self, content: str) -> str:
        """Extract MAINTAIN/REVISE/ABSTAIN from the response."""
        import re
        match = re.search(r'立场[：:]\s*(MAINTAIN|REVISE|ABSTAIN)', content)
        if match:
            return match.group(1)
        # Fallback: check last line
        last = content.strip().split("\n")[-1].strip()
        for decision in ("MAINTAIN", "REVISE", "ABSTAIN"):
            if decision in last:
                return decision
        return "ABSTAIN"
