"""Fail-closed API request, token, and estimated-cost protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from artifact_delib.api.schemas import TokenUsage


class BudgetExceeded(RuntimeError):
    """Raised before a request that would exceed configured limits."""


@dataclass
class BudgetManager:
    """Track cumulative limits without performing any provider calls."""

    max_requests: int
    max_tokens: int
    max_cost: float
    requests: int = 0
    tokens: int = 0
    cost: float = 0.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reserve(self, usage: TokenUsage, estimated_cost: float | None) -> None:
        """Reserve a completed/estimated request or raise before state changes."""
        with self._lock:
            next_cost = self.cost + (estimated_cost or 0.0)
            if self.requests + 1 > self.max_requests:
                raise BudgetExceeded("API_MAX_TOTAL_REQUESTS exceeded")
            if self.tokens + usage.total_tokens > self.max_tokens:
                raise BudgetExceeded("API_MAX_TOTAL_TOKENS exceeded")
            if next_cost > self.max_cost:
                raise BudgetExceeded("API_MAX_ESTIMATED_COST exceeded")
            self.requests += 1
            self.tokens += usage.total_tokens
            self.cost = next_cost
