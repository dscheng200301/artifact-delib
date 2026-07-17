"""Narrow JSON parsing for requested structured provider responses."""

from __future__ import annotations

import json


class StructuredResponseError(ValueError):
    """Raised when a response cannot safely be interpreted as a JSON object."""


def parse_json_object(content: str) -> dict[str, object]:
    """Parse a JSON object, accepting only a single optional Markdown fence."""

    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise StructuredResponseError("response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise StructuredResponseError("response must be a JSON object")
    return parsed
