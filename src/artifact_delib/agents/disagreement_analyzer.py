"""Disagreement Analyzer — analyzes why Top-K candidates differ."""

from __future__ import annotations

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import CandidateSet, DisagreementAnalysis, SummarizedReport


class DisagreementAnalyzer(ArtifactAgent):
    """Analyze the disagreement pattern between Top-K candidates.

    Outputs NL analysis and a route hint for the router.
    """

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "disagreement_analyzer", model_name)

    def analyze(
        self,
        candidates: CandidateSet,
        summary: SummarizedReport,
    ) -> DisagreementAnalysis:
        system = (
            "你是古代文物候选分歧分析专家。你的任务是比较Top-K候选文物身份之间的差异，"
            "分析当前主要的不确定性来自哪个方面。\n\n"
            "重点判断分歧类型：\n"
            "- SHAPE：候选在器形方面存在差异\n"
            "- STYLE：候选在纹饰风格或年代特征上存在差异\n"
            "- GLYPH：候选差异与铭文款识的可辨认程度有关\n"
            "- MATERIAL：候选在材质或工艺判断上存在差异\n"
            "- LOCAL_DETAIL：关键区分信息需要观察局部细节\n"
            "- MULTI_FACTOR：多种因素共同导致不确定性\n\n"
            "输出一段自然语言分析，然后在最后一行输出'分歧类型：X'，其中X是以上类型之一。"
        )
        candidates_text = "\n".join(
            f"{i+1}. {c.text} (confidence: {c.confidence:.2f})"
            for i, c in enumerate(candidates.candidates)
        )
        user = (
            f"当前Top-K候选：\n{candidates_text}\n\n"
            f"综合视觉报告：\n{summary.content}\n\n"
            "请分析当前候选之间的主要分歧来源。"
        )
        content, usage = self._call(system, user, max_output_tokens=512)
        hint = self._extract_hint(content)
        return DisagreementAnalysis(content=content.strip(), route_hint=hint, usage=usage)

    def _extract_hint(self, content: str) -> str:
        """Extract the route hint from the analysis content."""
        import re
        match = re.search(r'分歧类型[：:]\s*(\S+)', content)
        if match:
            hint = match.group(1).strip().upper()
            valid = {"SHAPE", "STYLE", "GLYPH", "MATERIAL", "LOCAL_DETAIL", "MULTI_FACTOR"}
            if hint in valid:
                return hint  # type: ignore[return-value]
        return "MULTI_FACTOR"
