"""Small, explicit retry policy for transient API failures."""

from __future__ import annotations

from collections.abc import Callable

import httpx


def retry_call[T](
    operation: Callable[[], T],
    max_retries: int,
    on_error: Callable[[Exception, int], None] | None = None,
) -> T:
    """Retry timeouts and 429/5xx responses up to ``max_retries`` attempts."""

    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if on_error is not None:
                on_error(exc, attempt + 1)
            retryable = isinstance(exc, (TimeoutError, httpx.TimeoutException))
            if isinstance(exc, httpx.HTTPStatusError):
                retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
            if not retryable or attempt == attempts - 1:
                raise
    raise RuntimeError("retry loop exhausted unexpectedly")
