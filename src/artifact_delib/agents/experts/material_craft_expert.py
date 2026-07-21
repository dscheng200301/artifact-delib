"""Material & Craft Expert — analyzes visual material appearance and technique.

Supports recheck mode focused on distinguishing candidates via material evidence.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport


_MATERIAL_NORMAL = (
    "你是古代文物材质与工艺专家。你的任务是分析一张文物图片中可见的材质和工艺特征。\n\n"
    "重点分析视觉上可观察到的特征：\n"
    "- 材质表象（陶、瓷、青铜、玉、金银、漆器等）\n"
    "- 工艺痕迹（铸造、雕刻、烧制、施釉等）\n"
    "- 釉面状态（颜色、光泽、开片、积釉等）\n"
    "- 露胎处特征（胎色、胎质等）\n\n"
    "注意：\n"
    "- 不能从RGB图片推断化学成分、金属比例或胎土实验数据\n"
    "- 只能描述视觉上可见的特征\n\n"
    "输出100到200字的专业自然语言分析。\n"
    "不要输出JSON或结构化格式。"
)

_MATERIAL_RECHECK = (
    "你是古代文物材质与工艺专家——当前处于定向重审模式。\n\n"
    "你已看到下文中两个候选之间的差异说明。\n"
    "请重新仔细查看图片，重点分析材质和工艺特征中能够区分两个候选的视觉信息。\n\n"
    "具体要求：\n"
    "- 比较两个候选在材质表象、釉面状态上的可能差异\n"
    "- 分析釉色、光泽、胎体特征更接近哪一个候选\n"
    "- 如果材质特征可以提供区分信息，请明确指出方向\n"
    "- 输出100到200字的专业自然语言分析\n\n"
    "不要输出JSON或结构化格式。"
)


class MaterialCraftExpert(ArtifactAgent):
    """Analyze visually observable material and craft characteristics."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "material_expert", model_name)

    def analyze(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> ExpertReport:
        is_recheck = context is not None and ("候选" in context and "区分" in context)
        system = _MATERIAL_RECHECK if is_recheck else _MATERIAL_NORMAL
        user = "请分析这张文物图片中可见的材质和工艺特征。"
        if context:
            user += f"\n\n【重审上下文】\n{context}"
        content, usage = self._call(system, user, image_path)
        return ExpertReport(expert_name="material", content=content.strip(), usage=usage)
