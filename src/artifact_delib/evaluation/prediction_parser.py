"""EvaluationParser — extracts structured evaluation data from NL identification.

THIS MODULE IS FOR EVALUATION ONLY. It never participates in inference,
routing, recheck, deliberation, or judgment.

It parses the final NL identification text to extract:
- category (e.g., 瓷器, 青铜器, 玉器)
- fine_grained_type (e.g., 青花梅瓶, 鼎, 玉璧)
- period (e.g., 明永乐, 商代晚期, 战国)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedIdentification:
    """Structured prediction extracted from NL — only for evaluation."""

    category: str | None = None
    fine_grained_type: str | None = None
    period: str | None = None
    dynasty: str | None = None
    raw_text: str = ""


# ── Known artifact type keywords ──

_CATEGORY_KEYWORDS = {
    "瓷器": "瓷器",
    "陶瓷": "瓷器",
    "瓷": "瓷器",
    "青花": "瓷器",
    "青铜": "青铜器",
    "铜": "青铜器",
    "玉": "玉器",
    "漆": "漆器",
    "金银": "金银器",
    "陶": "陶器",
}

_TYPE_KEYWORDS = [
    "梅瓶", "玉壶春瓶", "瓶", "壶",
    "鼎", "簋", "爵", "尊", "彝",
    "盘", "碗", "杯", "盏",
    "罐", "缸",
    "玉璧", "玉琮", "玉璋", "玉圭", "玉璜",
    "俑", "佛像",
    "镜",
]

_PERIOD_PATTERNS = [
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
        return ParsedIdentification(
            category=category,
            fine_grained_type=fine_type,
            period=period,
            dynasty=dynasty,
            raw_text=text[:200],
        )

    def _extract_category(self, text: str) -> str | None:
        """Extract artifact category from NL text."""
        for keyword, category in _CATEGORY_KEYWORDS.items():
            if keyword in text:
                return category
        return None

    def _extract_type(self, text: str) -> str | None:
        """Extract fine-grained artifact type from NL text."""
        found = []
        for kw in _TYPE_KEYWORDS:
            if kw in text:
                found.append(kw)
        # Return the most specific match (longest keyword)
        if found:
            return max(found, key=len)
        return None

    def _extract_period_and_dynasty(
        self, text: str
    ) -> tuple[str | None, str | None]:
        """Extract period and dynasty from NL text."""
        for pattern in _PERIOD_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                # Remove duplicates while preserving order
                seen: set[str] = set()
                unique = []
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
        self, prediction: ParsedIdentification, gold: ParsedIdentification
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
            "joint_correct": False,  # computed below
        }
