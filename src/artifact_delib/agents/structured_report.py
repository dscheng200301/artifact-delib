"""Structured expert report parsing — natural language + control fields.

Expert agents produce a natural language report as the primary output, with
optional lightweight structured fields embedded in (or alongside) the NL text.

Supported extraction strategies:
  1. JSON block in the output (```json { ... } ```)
  2. Structured fields appended after the NL report
  3. Pure NL fallback when neither is present

This module implements fault-tolerant parsing that always preserves the
original NL content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── Max candidates in structured field ──
MAX_CANDIDATES = 3
VALID_EXPERTS = frozenset([
    "shape", "style", "glyph", "material", "local_detail",
])


@dataclass
class StructuredReport:
    """Parsed structured report from an expert agent.

    The `report` field always contains the full natural-language analysis.
    All other fields are optional control hints for routing / disagreement analysis.
    """

    expert_type: str = ""
    report: str = ""
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_focus: list[str] = field(default_factory=list)
    recommended_expert: str = ""

    # Raw text that was parsed (for provenance)
    raw_text: str = ""

    # Whether structured parsing succeeded
    has_control_fields: bool = False


# ── Regex patterns ──

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?({.*?})\n?\s*```",
    re.DOTALL,
)

_CANDIDATE_BLOCK_RE = re.compile(
    r"(?:候选|candidates?|top candidates?)[：:]\s*(.+?)(?:\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"(?:不确定|uncertainty|需要进一步确认|重点关注)[：:]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

_RECOMMENDED_EXPERT_RE = re.compile(
    r"(?:建议|recommend(?:ed)?|下一个专家)[：:]\s*(\w+)",
    re.IGNORECASE,
)


def parse_expert_response(
    raw_text: str,
    expert_type: str = "",
) -> StructuredReport:
    """Parse a raw expert response into a StructuredReport.

    Tries in order:
    1. Extract JSON block from markdown code fence
    2. Extract NL + inline structured annotations
    3. Pure NL fallback

    The `report` field always holds the full NL content.
    """
    report = StructuredReport(
        expert_type=expert_type,
        raw_text=raw_text,
        report=raw_text,
    )

    text = raw_text.strip()
    if not text:
        return report

    # ── Strategy 1: JSON block ──
    parsed = _try_parse_json_block(text)
    if parsed is not None:
        report.has_control_fields = True
        # Preserve NL text OUTSIDE the JSON block as the main report.
        # Fall back to JSON's "report" field only if outside text is empty.
        outside_text = _strip_json_block(text).strip()
        if outside_text:
            report.report = outside_text
        else:
            report.report = parsed.get("report", text) or text

        candidates_raw = parsed.get("top_candidates", [])
        if isinstance(candidates_raw, list):
            valid = []
            for c in candidates_raw[:MAX_CANDIDATES]:
                if isinstance(c, dict) and "name" in c:
                    confidence = c.get("confidence", 0.0)
                    confidence = _clamp_confidence(confidence)
                    valid.append({"name": c["name"], "confidence": confidence})
            report.top_candidates = valid

        uncertainty = parsed.get("uncertainty_focus", [])
        if isinstance(uncertainty, list):
            report.uncertainty_focus = [str(u) for u in uncertainty]

        recommended = parsed.get("recommended_expert", "")
        if isinstance(recommended, str) and recommended.lower() in VALID_EXPERTS:
            report.recommended_expert = recommended.lower()

        return report

    # ── Strategy 2: Inline annotations ──
    report.has_control_fields = _try_extract_inline_fields(report)

    return report


def _try_parse_json_block(text: str) -> dict[str, Any] | None:
    """Try to extract and parse a JSON block from the text."""
    # Look for JSON code fence first
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Try direct JSON parse of the whole text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _strip_json_block(text: str) -> str:
    """Remove JSON code blocks from text, returning the surrounding NL."""
    return _JSON_BLOCK_RE.sub("", text).strip()


def _try_extract_inline_fields(report: StructuredReport) -> bool:
    """Try to extract structured fields from NL text annotations.

    Returns True if any field was found.
    """
    text = report.raw_text
    found_any = False

    # Candidates from inline text
    # Look for lines like: "候选：宋青瓷碗(0.67)"
    candidate_lines = _CANDIDATE_BLOCK_RE.findall(text)
    if candidate_lines:
        candidates = []
        for line in candidate_lines:
            items = re.findall(
                r"(.+?)\((\d+\.?\d*)\)",
                line,
            )
            for name, conf_str in items:
                name = name.strip()
                confidence = _clamp_confidence(float(conf_str))
                if name:
                    candidates.append({"name": name, "confidence": confidence})
        if candidates:
            report.top_candidates = candidates[:MAX_CANDIDATES]
            found_any = True

    # Uncertainty focus
    uncertainty_matches = _UNCERTAINTY_RE.findall(text)
    if uncertainty_matches:
        focus_items = []
        for match in uncertainty_matches:
            items = re.split(r"[，,、\s]+", match.strip())
            focus_items.extend(items)
        if focus_items:
            report.uncertainty_focus = [f.strip() for f in focus_items if f.strip()]
            found_any = True

    # Recommended expert
    expert_match = _RECOMMENDED_EXPERT_RE.search(text)
    if expert_match:
        expert = expert_match.group(1).lower().strip()
        if expert in VALID_EXPERTS:
            report.recommended_expert = expert
            found_any = True

    return found_any


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def reconstruct_report(report: StructuredReport) -> str:
    """Reconstruct a combined output from structured fields + NL report.

    If the original had a JSON block, strips it from the NL and returns
    clean NL. Otherwise returns the original NL.
    """
    text = report.raw_text
    if report.has_control_fields:
        # Strip JSON block if present
        cleaned = _JSON_BLOCK_RE.sub("", text).strip()
        return cleaned or report.report
    return text
