"""Shape Expert — analyzes form, proportion, structural features.

Supports recheck mode: when context contains candidate comparison info,
uses a recheck-specific prompt focused on distinguishing candidates.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport


_SHAPE_NORMAL = (
    "你是古代文物器形专家。你的任务是分析一张文物图片中的器形特征。\n\n"
    "重点分析：\n"
    "- 整体器形类别（鼎、簋、爵、尊、瓶、梅瓶、玉壶春瓶、罐、盘、碗等）\n"
    "- 口沿、颈部、肩部、腹部、底部、足部的形态\n"
    "- 器物各部位的比例关系\n"
    "- 耳、柄、足等附件结构\n\n"
    "输出100到200字的专业自然语言分析。\n"
    "不要输出JSON或结构化格式。\n"
    "不要直接断定文物年代或具体名称——只描述器形特征。"
)

_SHAPE_RECHECK = (
    "你是古代文物器形专家——当前处于定向重审模式。\n\n"
    "你已看到下文中两个候选之间的差异说明。\n"
    "请重新仔细查看图片，重点关注那些能够区分两个候选的器形细节。\n\n"
    "具体要求：\n"
    "- 比较两个候选在器形上的可能差异点\n"
    "- 说明器形特征更支持哪一个候选，并给出理由\n"
    "- 如果器形特征无法区分，也如实说明\n"
    "- 输出100到200字的专业自然语言分析\n\n"
    "不要输出JSON或结构化格式。"
)


class ShapeExpert(ArtifactAgent):
    """Analyze artifact shape: form, proportion, mouth/neck/shoulder/belly/foot."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "shape_expert", model_name)

    def analyze(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> ExpertReport:
        is_recheck = context is not None and ("候选" in context and "区分" in context)
        system = _SHAPE_RECHECK if is_recheck else _SHAPE_NORMAL
        user = "请分析这张文物图片的器形特征。"
        if context:
            user += f"\n\n【重审上下文】\n{context}"
        content, usage = self._call(system, user, image_path)
        return ExpertReport(expert_name="shape", content=content.strip(), usage=usage)
