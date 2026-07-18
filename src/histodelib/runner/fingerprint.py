"""Deterministic run identity for safe resumability."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_run_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 identity for all run-defining inputs."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
