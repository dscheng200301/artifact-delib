"""Deferred Artifact Judge — final natural-language identification.

The judge is the LAST module called. It reviews the original image,
all expert reports, Top-K candidates, any recheck results, and any
deliberation summary before producing a final NL identification.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import (
    CandidateSet,
    DeliberationResult,
    ExpertReport,
    FinalIdentification,
    SummarizedReport,
    VisualPerceptionReport,
)


class ArtifactJudge(ArtifactAgent):
    """Deferred artifact judge — final NL identification after all evidence is gathered."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "judge", model_name)

    def adjudicate(
        self,
        image_path: Path,
        visual_report: VisualPerceptionReport,
        summarized_report: SummarizedReport,
        candidates: CandidateSet,
        expert_reports: tuple[ExpertReport, ...] = (),
        recheck_reports: tuple[ExpertReport, ...] = (),
        deliberation_result: DeliberationResult | None = None,
    ) -> FinalIdentification:
        system = (
            "你是古代文物最终鉴定专家。你的任务是基于所有可获得的信息，"
            "输出最终的自然语言文物识别结果。\n\n"
            "要求：\n"
            "- 输出一段完整的自然语言段落\n"
            "- 包含：可能的文物大类、细粒度类型、年代/朝代判断\n"
            "- 有充分把握时给出地域、窑口、文化体系推测\n"
            "- 存在不确定性时如实说明局限\n"
            "- 不输出JSON或其他结构化格式\n\n"
            "请重新查看原始图片，综合考虑所有专家的分析后再做判断。"
        )
        evidence = self._build_evidence(
            summarized_report, candidates, recheck_reports, deliberation_result
        )
        content, usage = self._call(system, evidence, image_path, max_output_tokens=512)
        return FinalIdentification(content=content.strip(), usage=usage)

    def _build_evidence(
        self,
        summarized: SummarizedReport,
        candidates: CandidateSet,
        recheck_reports: tuple[ExpertReport, ...],
        deliberation: DeliberationResult | None,
    ) -> str:
        parts = [f"综合视觉分析报告：\n{summarized.content}"]
        candidates_text = "\n".join(
            f"{i+1}. {c.text} (confidence: {c.confidence:.2f})"
            for i, c in enumerate(candidates.candidates)
        )
        parts.append(f"\n\nTop-K候选：\n{candidates_text}")
        if recheck_reports:
            recheck_text = "\n\n".join(
                f"【{r.expert_name}重审】{r.content}" for r in recheck_reports
            )
            parts.append(f"\n\n定向重审结果：\n{recheck_text}")
        if deliberation is not None:
            parts.append(f"\n\n协商记录：{deliberation.summary}")
        parts.append("\n\n请基于以上所有信息，结合你对原始图片的观察，输出最终的自然语言文物识别结果。")
        return "\n".join(parts)
