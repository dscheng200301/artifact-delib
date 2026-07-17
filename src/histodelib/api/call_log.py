"""Redacted JSONL audit log for API calls."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SECRET = re.compile(r"(?i)(authorization\s*:\s*bearer\s+|api[_-]?key\s*[=:]\s*)([^\s,;]+)")


def redact_secrets(value: str) -> str:
    """Remove bearer tokens and API-key values from diagnostic text."""

    return _SECRET.sub(r"\1***REDACTED***", value)


class CallLogStore:
    """Append sanitized call summaries without storing raw credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        clean = {
            key: redact_secrets(str(value)) if isinstance(value, str) else value
            for key, value in record.items()
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
