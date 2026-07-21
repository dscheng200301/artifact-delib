"""Visual Perception Agent — describes what's visible without identifying."""

from __future__ import annotations

from pathlib import Path

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import VisualPerceptionReport


class VisualPerceptionAgent(ArtifactAgent):
    """Describe the overall visible content of an artifact image.

    Outputs a brief NL observation (100-200 chars) capturing what is seen,
    without jumping to identification conclusions.
    """

    def __init__(self, client, model_name: str = "default") -> None:
        super().__init__(client, "visual_perception", model_name)

    def analyze(self, image_path: Path) -> VisualPerceptionReport:
        system = (
            "你是古代文物视觉观察专家。你的任务是观察一张文物图片，"
            "用100到200字描述你整体看到了什么。\n\n"
            "只描述视觉特征：器形轮廓、颜色、纹理、表面状态、可见结构等。\n"
            "不要直接判断文物的具体名称、年代或真伪。\n"
            "不要输出JSON或其他结构化格式。\n"
            "直接输出一段流畅的自然语言描述。"
        )
        user = "请仔细观察这张文物图片，描述你整体看到的视觉特征。"
        content, usage = self._call(system, user, image_path)
        return VisualPerceptionReport(content=content.strip(), usage=usage)
