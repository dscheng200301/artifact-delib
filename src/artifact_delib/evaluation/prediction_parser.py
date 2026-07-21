"""EvaluationParser — extracts structured evaluation data from NL identification.

THIS MODULE IS FOR EVALUATION ONLY. It never participates in inference,
routing, recheck, deliberation, or judgment.

It parses the final NL identification text to extract:
- category (e.g., 瓷器, 青铜器, 玉器)
- fine_grained_type (e.g., 青花梅瓶, 鼎, 玉璧)
- period (e.g., 明永乐, 商代晚期, 战国)
- material (e.g., 瓷, 青铜, 玉, 漆, 金银, 陶)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedIdentification:
    """Structured prediction extracted from NL — only for evaluation."""

    category: str | None = None
    fine_grained_type: str | None = None
    period: str | None = None
    dynasty: str | None = None
    material: str | None = None
    raw_text: str = ""


# ── Category keywords ──

_CATEGORY_KEYWORDS: dict[str, str] = {
    "瓷器": "瓷器",
    "陶瓷": "瓷器",
    "青花": "瓷器",
    "青铜": "青铜器",
    "铜器": "青铜器",
    "玉器": "玉器",
    "漆器": "漆器",
    "金银": "金银器",
    "金器": "金银器",
    "银器": "金银器",
    "陶器": "陶器",
    "珐琅": "珐琅器",
    "景泰蓝": "珐琅器",
    "雕塑": "雕塑",
    "石刻": "雕塑",
    "石雕": "雕塑",
    "木雕": "雕塑",
    "纺织品": "纺织品",
    "织物": "纺织品",
    "刺绣": "纺织品",
    "缂丝": "纺织品",
    "书法": "书法",
    "绘画": "绘画",
    "画像": "绘画",
}


# ── Material keywords (mapped to canonical names) ──

_MATERIAL_KEYWORDS: dict[str, str] = {
    # Ceramic / Porcelain
    "瓷质": "瓷",
    "瓷器": "瓷",
    "瓷胎": "瓷",
    "青花瓷": "瓷",
    "白瓷": "瓷",
    "青瓷": "瓷",
    "青白瓷": "瓷",
    "粉彩": "瓷",
    "五彩": "瓷",
    "斗彩": "瓷",
    "釉里红": "瓷",
    "瓷": "瓷",
    # Bronze
    "青铜": "青铜",
    "铜质": "青铜",
    "铜": "青铜",
    # Jade
    "玉质": "玉",
    "玉石": "玉",
    "白玉": "玉",
    "青玉": "玉",
    "碧玉": "玉",
    "黄玉": "玉",
    "玉": "玉",
    # Lacquer
    "漆质": "漆",
    "漆器": "漆",
    "漆": "漆",
    # Gold/Silver
    "金质": "金银",
    "银质": "金银",
    "鎏金": "金银",
    "镀金": "金银",
    "金银": "金银",
    "金": "金银",
    "银": "金银",
    # Pottery
    "陶质": "陶",
    "陶器": "陶",
    "陶": "陶",
    # Enamel
    "珐琅": "珐琅",
    "景泰蓝": "珐琅",
    "掐丝珐琅": "珐琅",
    "画珐琅": "珐琅",
    # Stone
    "石质": "石",
    "石材": "石",
    "寿山石": "石",
    "田黄": "石",
    "大理石": "石",
    "石": "石",
    # Wood
    "木质": "木",
    "木材": "木",
    "紫檀": "木",
    "黄花梨": "木",
    "楠木": "木",
    "木": "木",
    # Textile
    "丝质": "丝",
    "绢": "丝",
    "丝绸": "丝",
    "织物": "丝",
    "丝": "丝",
    "缎": "丝",
    "锦": "丝",
    # Bone/Ivory
    "象牙": "牙骨",
    "骨": "牙骨",
    "犀角": "牙骨",
    # Glass
    "玻璃": "玻璃",
    "琉璃": "玻璃",
    "料器": "玻璃",
    # Mixed
    "铜鎏金": "铜鎏金",
}


# ── Artifact type keywords ──

_TYPE_KEYWORDS: list[str] = [
    "玉壶春瓶", "梅瓶", "天球瓶", "胆瓶", "蒜头瓶",
    "瓶", "壶",
    "鼎", "簋", "爵", "尊", "彝", "觚", "斝", "卣",
    "盘", "碗", "杯", "盏", "碟", "钵",
    "罐", "缸", "瓮",
    "玉璧", "玉琮", "玉璋", "玉圭", "玉璜", "玉玦",
    "俑", "佛像", "菩萨像", "天王像",
    "镜", "炉", "灯", "洗", "砚", "枕", "盒",
    "如意", "香炉", "笔筒", "笔洗", "水盂",
    "屏风", "挂轴", "扇面", "册页",
]


# ── Period patterns ──

_PERIOD_PATTERNS: list[str] = [
    # 完整时期: 明代永乐时期, 清代康熙年间, 商代晚期
    r"((?:西|东)?[秦汉唐宋元明清商][代朝]?(?:[^\s，。]{1,3})?(?:时期|年间|早期|中期|晚期))",
    # 短朝代+年号/时期名: 明永乐, 清康熙, 明宣德
    r"((?:西|东)?[秦汉唐宋元明清商][代朝]?[^\s，。]{1,3}(?=青花|瓷器|铜器|玉器|[，。\s]))",
    # 单独朝代+时期: 商代, 西周, 春秋, 战国, 西汉
    r"((?:夏|商|周|西周|东周|春秋|战国|秦|汉|西汉|东汉|三国|晋|南北朝|隋|唐|五代|宋|辽|金|元|明|清)(?:代|朝|时期)?)",
]


class PredictionParser:
    """Parse NL identification text into structured evaluation data.

    Thread-safe and stateless — one instance can be reused.
    """

    def parse(self, text: str) -> ParsedIdentification:
        """Parse a single NL identification into structured fields."""
        category = self._extract_category(text)
        fine_type = self._extract_type(text)
        period, dynasty = self._extract_period_and_dynasty(text)
        material = self._extract_material(text)
        return ParsedIdentification(
            category=category,
            fine_grained_type=fine_type,
            period=period,
            dynasty=dynasty,
            material=material,
            raw_text=text[:200],
        )

    def _extract_category(self, text: str) -> str | None:
        """Extract artifact category from NL text."""
        # Use longest-match-first to prefer specific over generic
        sorted_keywords = sorted(_CATEGORY_KEYWORDS, key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in text:
                return _CATEGORY_KEYWORDS[keyword]
        return None

    def _extract_type(self, text: str) -> str | None:
        """Extract fine-grained artifact type from NL text.

        Returns the most specific match (longest keyword).
        """
        found: list[str] = []
        for kw in _TYPE_KEYWORDS:
            if kw in text:
                found.append(kw)
        if found:
            return max(found, key=len)
        return None

    def _extract_material(self, text: str) -> str | None:
        """Extract artifact material from NL text.

        Uses longest-match-first to prefer specific material descriptions.
        """
        sorted_keywords = sorted(_MATERIAL_KEYWORDS, key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in text:
                return _MATERIAL_KEYWORDS[keyword]
        return None

    def _extract_period_and_dynasty(
        self, text: str
    ) -> tuple[str | None, str | None]:
        """Extract period and dynasty from NL text."""
        for pattern in _PERIOD_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                seen: set[str] = set()
                unique: list[str] = []
                for m in matches:
                    m = m.strip()
                    if m and m not in seen:
                        seen.add(m)
                        unique.append(m)
                if unique:
                    period = unique[0]
                    dynasty = self._extract_dynasty_from_period(period)
                    return period, dynasty
        return None, None

    @staticmethod
    def _extract_dynasty_from_period(period: str) -> str | None:
        """Extract the dynasty name from a period string."""
        dynasty_keywords = [
            "夏", "商", "周", "西周", "东周", "春秋", "战国",
            "秦", "汉", "西汉", "东汉", "三国", "晋",
            "南北朝", "隋", "唐", "五代",
            "宋", "辽", "金", "元", "明", "清",
        ]
        for dk in dynasty_keywords:
            if dk in period:
                return dk
        return None

    def evaluate(
        self,
        prediction: ParsedIdentification,
        gold: ParsedIdentification,
    ) -> dict[str, bool]:
        """Compare a parsed prediction against gold labels."""
        return {
            "category_correct": (
                prediction.category == gold.category
                if prediction.category and gold.category
                else False
            ),
            "type_correct": (
                prediction.fine_grained_type == gold.fine_grained_type
                if prediction.fine_grained_type and gold.fine_grained_type
                else False
            ),
            "period_correct": (
                prediction.period == gold.period
                if prediction.period and gold.period
                else False
            ),
            "material_correct": (
                prediction.material == gold.material
                if prediction.material and gold.material
                else False
            ),
            "joint_correct": False,  # computed below
        }
