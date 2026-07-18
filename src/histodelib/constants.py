"""Stable project defaults used by API-only runs."""

DEFAULT_MODEL = "qwen3.5-flash-2026-02-23"
JSON_RESPONSE_SCHEMA = {"type": "json_object"}
LABEL_JSON_INSTRUCTION = (
    'Return only valid JSON with a "label" key whose value is exactly '
    '"TRUE", "MISCAPTIONED", or "OUT_OF_CONTEXT", plus concise evidence.'
)
