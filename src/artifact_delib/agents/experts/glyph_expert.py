"""Glyph Expert — analyzes inscriptions, marks, seals, reign marks.

Supports recheck mode focused on distinguishing candidates via textual content.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport


_GLYPH_NORMAL = (
    "你是古代文物铭文款识专家。你的任务是分析一张文物图片中可能存在的文字信息。\n\n"
    "重点分析：\n"
    "- 铭文、款识、印章、题跋、年号等文字区域\n"
    "- 文字布局（位置、行数、排列方式）\n"
    "- 文字可辨认程度\n\n"
    "注意：\n"
    "- 看到'大明宣德年制'不能直接判断器物一定是明宣德真品\n"
    "- 款识内容本身不能证明真实年代，只能说明器物上存在什么文字\n"
    "- 说明当前图像质量和角度对文字辨认的影响\n\n"
    "输出100到200字的专业自然语言分析。\n"
    "不要输出JSON或结构化格式。"
)

_GLYPH_RECHECK = (
    "你是古代文物铭文款识专家——当前处于定向重审模式。\n\n"
    "你已看到下文中两个候选之间的差异说明。\n"
    "请重新仔细查看图片，重点寻找能够区分两个候选的铭文或款识信息。\n\n"
    "具体要求：\n"
    "- 重新检查器底、器身、口沿等所有可能存在文字的区域\n"
    "- 比较两个候选在款识内容、位置、字体风格上的可能差异\n"
    "- 如果可以辨认部分文字，说明其与哪个候选更吻合\n"
    "- 如果不能辨认，说明款识区域的存在本身是否支持某个候选\n"
    "- 输出100到200字的专业自然语言分析\n\n"
    "不要输出JSON或结构化格式。"
)


class GlyphExpert(ArtifactAgent):
    """Analyze visible inscriptions, marks, seals, and textual content."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "glyph_expert", model_name)

    def analyze(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> ExpertReport:
        is_recheck = context is not None and ("候选" in context and "区分" in context)
        system = _GLYPH_RECHECK if is_recheck else _GLYPH_NORMAL
        user = "请分析这张文物图片中的铭文、款识或文字信息。"
        if context:
            user += f"\n\n【重审上下文】\n{context}"
        content, usage = self._call(system, user, image_path)
        return ExpertReport(expert_name="glyph", content=content.strip(), usage=usage)
