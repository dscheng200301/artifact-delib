"""Style Expert — analyzes decoration, pattern, composition, period style.

Supports recheck mode focused on distinguishing candidate periods.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport


_STYLE_NORMAL = (
    "你是古代文物纹饰与艺术风格专家。你的任务是分析一张文物图片中的装饰风格。\n\n"
    "重点分析：\n"
    "- 纹饰类型（兽面纹、夔龙纹、云雷纹、缠枝莲、龙凤纹、山水纹等）\n"
    "- 构图布局和装饰分区\n"
    "- 色彩表现（青花发色、釉色、彩绘等）\n"
    "- 整体艺术风格和时代视觉特征\n\n"
    "输出100到200字的专业自然语言分析。\n"
    "不要输出JSON或结构化格式。\n"
    "不要直接断定文物具体年代——只描述纹饰和风格特征。"
)

_STYLE_RECHECK = (
    "你是古代文物纹饰与艺术风格专家——当前处于定向重审模式。\n\n"
    "你已看到下文中两个候选之间的年代或风格差异说明。\n"
    "请重新仔细查看图片，重点分析纹饰中能够区分两个候选年代的特征。\n\n"
    "具体要求：\n"
    "- 比较两个候选在纹饰、构图、色彩上的可能差异\n"
    "- 分析纹饰风格更接近哪一个时期或类型，并给出具体理由\n"
    "- 特别注意：纹饰布局的疏密、笔触的力度、色彩的深浅\n"
    "- 输出100到200字的专业自然语言分析\n\n"
    "不要输出JSON或结构化格式。"
)


class StyleExpert(ArtifactAgent):
    """Analyze decorative style, patterns, composition, and period visual cues."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "style_expert", model_name)

    def analyze(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> ExpertReport:
        is_recheck = context is not None and ("候选" in context and "区分" in context)
        system = _STYLE_RECHECK if is_recheck else _STYLE_NORMAL
        user = "请分析这张文物图片的纹饰和艺术风格特征。"
        if context:
            user += f"\n\n【重审上下文】\n{context}"
        content, usage = self._call(system, user, image_path)
        return ExpertReport(expert_name="style", content=content.strip(), usage=usage)
