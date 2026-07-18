"""Stable project defaults used by API-only runs."""

DEFAULT_MODEL = "qwen3.5-flash-2026-02-23"
JSON_RESPONSE_SCHEMA = {"type": "json_object"}
LABEL_JSON_INSTRUCTION = (
    'Return only valid JSON with a "label" key whose value is exactly '
    '"TRUE", "MISCAPTIONED", or "OUT_OF_CONTEXT", plus concise evidence.'
)
EVIDENCE_JSON_INSTRUCTION = (
    "Return only valid JSON with modality-specific evidence fields, uncertainty, and an "
    "optional candidate_label (TRUE, MISCAPTIONED, or OUT_OF_CONTEXT). Do not emit a "
    "cross-modal final decision."
)
