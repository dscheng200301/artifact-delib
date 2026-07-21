"""Credential-free, local JSON cache for normalized responses."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from filelock import FileLock

T = TypeVar("T")


class ResponseCache:
    """Store serializable values under caller-provided non-secret cache keys."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_set(self, key: str, producer: Callable[[], T]) -> tuple[T, str]:
        """Return a cached JSON value or atomically persist a newly produced one."""

        if not key.replace("-", "").replace("_", "").isalnum():
            raise ValueError("cache key must be alphanumeric, hyphen, or underscore")
        target = self.root / f"{key}.json"
        lock = FileLock(str(target.with_suffix(".lock")))
        with lock:
            if target.exists():
                return json.loads(target.read_text(encoding="utf-8")), "hit"
            value = producer()
            target.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            return value, "miss"
