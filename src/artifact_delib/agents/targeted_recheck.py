"""Targeted Expert Recheck — focused re-examination to resolve candidate disagreement.

This is the core module for Innovation 2: targeted recheck driven by candidate disagreement.
Instead of re-running all experts, it calls the single most relevant expert with a
context-rich prompt focused on distinguishing the top candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_delib.agents.experts.glyph_expert import GlyphExpert
from artifact_delib.agents.experts.local_detail_expert import LocalDetailExpert
from artifact_delib.agents.experts.material_craft_expert import MaterialCraftExpert
from artifact_delib.agents.experts.shape_expert import ShapeExpert
from artifact_delib.agents.experts.style_expert import StyleExpert
from artifact_delib.schemas import (
    CandidateSet,
    DisagreementAnalysis,
    ExpertReport,
    RecheckRecord,
    RouteDecision,
)

# Map route action → expert instance attribute name
ACTION_TO_EXPERT_ATTR = {
    "SHAPE_RECHECK": "shape_expert",
    "STYLE_RECHECK": "style_expert",
    "GLYPH_RECHECK": "glyph_expert",
    "MATERIAL_RECHECK": "material_expert",
    "LOCAL_DETAIL_RECHECK": "local_detail_expert",
}

# Map route action → human-readable Chinese expert name
ACTION_TO_EXPERT_NAME = {
    "SHAPE_RECHECK": "器形",
    "STYLE_RECHECK": "纹饰风格",
    "GLYPH_RECHECK": "铭文款识",
    "MATERIAL_RECHECK": "材质工艺",
    "LOCAL_DETAIL_RECHECK": "局部细节",
}

# Each expert's distinguishing specialty — used to craft the recheck question
_SPECIALTY_HINTS = {
    "SHAPE_RECHECK": (
        "器形差异：{c1}和{c2}在器形特征上可能存在差异。"
        "请重点观察口沿、颈部、肩部、腹部和底足的形态细节，"
        "寻找能够区分两个候选的器形特征。"
    ),
    "STYLE_RECHECK": (
        "纹饰风格差异：{c1}和{c2}在纹饰或年代风格上可能存在差异。"
        "请重点分析纹饰布局、图案细节、色彩表现和整体艺术风格，"
        "寻找能够区分两个候选的装饰特征。"
    ),
    "GLYPH_RECHECK": (
        "铭文款识差异：{c1}和{c2}在铭文或款识方面可能存在差异。"
        "请重新仔细检查器物上所有可能存在的文字、款识、印章或题跋，"
        "记录任何可辨认的内容及其位置特征。"
    ),
    "MATERIAL_RECHECK": (
        "材质工艺差异：{c1}和{c2}在材质或工艺特征上可能存在差异。"
        "请重点观察釉面状态、胎体特征、工艺痕迹和材质表象，"
        "寻找能够区分两个候选的材质或工艺特征。"
    ),
    "LOCAL_DETAIL_RECHECK": (
        "局部细节差异：{c1}和{c2}的关键区分信息可能隐藏在局部细节中。"
        "请重点观察器底、口沿、足部、接缝、磨损痕迹等容易被忽视的细节，"
        "寻找能够区分两个候选的关键细节特征。"
    ),
}


class TargetedExpertRecheck:
    """Coordinate targeted recheck of a single expert to resolve candidate disagreement.

    Features:
    - Rich context building (candidates + disagreement + recheck history)
    - Per-expert recheck prompts focused on discrimination
    - Version tracking (record what changed from previous report)
    """

    def __init__(
        self,
        shape_expert: ShapeExpert,
        style_expert: StyleExpert,
        glyph_expert: GlyphExpert,
        material_expert: MaterialCraftExpert,
        local_detail_expert: LocalDetailExpert,
    ) -> None:
        self._experts = {
            "shape_expert": shape_expert,
            "style_expert": style_expert,
            "glyph_expert": glyph_expert,
            "material_expert": material_expert,
            "local_detail_expert": local_detail_expert,
        }

    def execute(
        self,
        image_path: Path,
        route: RouteDecision,
        candidates: CandidateSet,
        disagreement: DisagreementAnalysis | None,
        current_reports: tuple[ExpertReport, ...],
        recheck_history: tuple[RecheckRecord, ...],
        round_no: int,
    ) -> RecheckRecord:
        """Execute one targeted recheck and return a version-tracked record."""
        action = route.action
        if action not in ACTION_TO_EXPERT_ATTR:
            raise ValueError(f"Unknown recheck action: {action}")

        attr = ACTION_TO_EXPERT_ATTR[action]
        expert = self._get_expert(attr)
        expert_name = ACTION_TO_EXPERT_NAME[action]

        # Build the context-rich recheck prompt
        context = self._build_context(action, candidates, disagreement, recheck_history)

        # Find the previous report for version tracking
        prev_content = self._find_previous_report(expert_name, current_reports)

        # Call the expert with recheck context
        new_report = expert.analyze(image_path, context=context)

        return RecheckRecord(
            round_no=round_no,
            expert_name=expert_name,
            previous_content=prev_content,
            new_content=new_report.content,
            context_query=context,
            usage=new_report.usage,
        )

    def _build_context(
        self,
        action: str,
        candidates: CandidateSet,
        disagreement: DisagreementAnalysis | None,
        history: tuple[RecheckRecord, ...],
    ) -> str:
        """Build a rich context string for the recheck."""
        parts: list[str] = []

        # 1. Candidate comparison
        c1 = candidates.top1.text if candidates.top1 else "未知"
        c2 = candidates.top2.text if candidates.top2 else "未知"
        parts.append(
            f"当前两个主要候选需要进一步区分：\n"
            f"  候选1（置信度{candidates.top1_confidence:.2f}）：{c1}\n"
            f"  候选2（置信度{candidates.top2_confidence:.2f}）：{c2}\n"
        )

        # 2. Disagreement analysis
        if disagreement is not None:
            parts.append(f"分歧分析：{disagreement.content}\n")

        # 3. Recheck history summary
        if history:
            history_lines = [f"  第{r.round_no}轮：{r.expert_name}专家重审" for r in history]
            parts.append("已有重审记录：\n" + "\n".join(history_lines) + "\n")

        # 4. Targeted question for this expert
        hint = _SPECIALTY_HINTS.get(action, "")
        if hint:
            parts.append(hint.format(c1=c1, c2=c2))

        return "\n".join(parts)

    def _find_previous_report(
        self, expert_name: str, current_reports: tuple[ExpertReport, ...]
    ) -> str:
        """Find the current report content for the given expert."""
        for report in current_reports:
            cn = {
                "器形": "shape",
                "纹饰风格": "style",
                "铭文款识": "glyph",
                "材质工艺": "material",
                "局部细节": "local_detail",
            }.get(expert_name, "")
            if report.expert_name == cn:
                return report.content
        return ""

    def _get_expert(self, attr_name: str) -> Any:
        """Get the expert instance by attribute name."""
        expert = self._experts.get(attr_name)
        if expert is None:
            raise ValueError(f"Unknown expert: {attr_name}")
        return expert
