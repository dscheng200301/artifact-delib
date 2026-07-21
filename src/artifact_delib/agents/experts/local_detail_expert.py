"""Local Detail Expert — analyzes specific regions for fine-grained details.

Supports recheck mode focused on distinguishing candidates via local features.
"""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ExpertReport


_LOCAL_NORMAL = (
    "你是古代文物局部细节专家。你的任务是分析一张文物图片中的局部细节特征。\n\n"
    "重点分析：\n"
    "- 器底、口沿、足部细节\n"
    "- 接缝、修坯痕迹\n"
    "- 局部纹样精细特征\n"
    "- 磨损、裂纹、修补痕迹\n"
    "- 釉面局部异常\n\n"
    "输出100到200字的专业自然语言分析。\n"
    "不要输出JSON或结构化格式。"
)

_LOCAL_RECHECK = (
    "你是古代文物局部细节专家——当前处于定向重审模式。\n\n"
    "你已看到下文中两个候选之间的差异说明。\n"
    "请重新仔细查看图片，重点关注那些容易被忽视但能够区分两个候选的局部细节。\n\n"
    "具体要求：\n"
    "- 仔细检查器底、足部、口沿、接缝等关键部位\n"
    "- 寻找能够区分两个候选的微观细节特征\n"
    "- 例如：圈足形态、修坯痕迹、釉面局部变化等\n"
    "- 输出100到200字的专业自然语言分析\n\n"
    "不要输出JSON或结构化格式。"
)


class LocalDetailExpert(ArtifactAgent):
    """Analyze local/specific regions: base, rim, foot, joints, damage."""

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "local_detail_expert", model_name)

    def analyze(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> ExpertReport:
        is_recheck = context is not None and ("候选" in context and "区分" in context)
        system = _LOCAL_RECHECK if is_recheck else _LOCAL_NORMAL
        user = "请分析这张文物图片的局部细节特征。"
        if context:
            user += f"\n\n【重审上下文】\n{context}"
        content, usage = self._call(system, user, image_path)
        return ExpertReport(expert_name="local_detail", content=content.strip(), usage=usage)
