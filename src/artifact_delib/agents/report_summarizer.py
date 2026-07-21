"""Expert Report Summarizer — compresses multi-expert reports into one NL summary."""

from __future__ import annotations

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport, SummarizedReport, VisualPerceptionReport


class ReportSummarizer(ArtifactAgent):
    """Compress multi-expert reports into a concise integrated visual analysis.

    Outputs 200-400 chars of NL.
    """

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "summarizer", model_name)

    def summarize(
        self,
        visual_report: VisualPerceptionReport,
        expert_reports: tuple[ExpertReport, ...],
    ) -> SummarizedReport:
        system = (
            "你是古代文物多专家分析综合员。你的任务是将多个专业视觉专家的分析报告"
            "整合为一段综合视觉分析。\n\n"
            "要求：\n"
            "- 压缩重复信息\n"
            "- 保留关键区分信息\n"
            "- 保留专家之间存在的不确定性\n"
            "- 输出200到400字的自然语言\n\n"
            "不要简单拼接所有专家文本。不要输出JSON或结构化格式。"
        )
        reports_text = "\n\n".join(
            f"【{r.expert_name}分析】{r.content}" for r in expert_reports
        )
        user = (
            f"【整体视觉观察】{visual_report.content}\n\n"
            f"【各专家分析】\n{reports_text}\n\n"
            "请整合以上所有分析，输出一段综合视觉分析报告。"
        )
        content, usage = self._call(system, user)
        return SummarizedReport(content=content.strip(), usage=usage)
