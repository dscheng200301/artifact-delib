"""Provider-neutral token and latency accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRecord:
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenAccounting:
    """Accumulate request usage for one run."""

    def __init__(self) -> None:
        self.records: list[TokenRecord] = []

    def record(self, request_id: str, input_tokens: int, output_tokens: int, latency_ms: float) -> TokenRecord:
        record = TokenRecord(request_id, input_tokens, output_tokens, latency_ms)
        self.records.append(record)
        return record

    @property
    def total_tokens(self) -> int:
        return sum(record.total_tokens for record in self.records)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def average_latency_ms(self) -> float:
        return sum(record.latency_ms for record in self.records) / len(self.records) if self.records else 0.0
