"""Provider-neutral audit wrapper for token, latency, cost, and error logging."""

from __future__ import annotations

from histodelib.api.base import ModelClient
from histodelib.api.call_log import CallLogStore
from histodelib.api.cost import estimate_cost
from histodelib.api.token_usage import TokenAccounting
from histodelib.schemas import ModelRequest, ModelResponse


class AuditedModelClient:
    """Wrap any model client and append one redacted record per attempted call."""

    def __init__(
        self,
        client: ModelClient,
        call_log: CallLogStore,
        pricing: dict[str, object] | None = None,
    ) -> None:
        self.client = client
        self.call_log = call_log
        self.pricing = pricing
        self.accounting = TokenAccounting()

    def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = self.client.generate(request)
        except Exception as exc:
            self.call_log.append(
                {
                    "request_id": request.request_id,
                    "provider": "unknown",
                    "model": request.model,
                    "cache_state": "disabled",
                    "error_type": type(exc).__name__,
                    "latency_ms": 0.0,
                }
            )
            raise

        record = self.accounting.record(
            request_id=response.request_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=response.latency_ms,
        )
        self.call_log.append(
            {
                "request_id": response.request_id,
                "provider": response.provider,
                "model": response.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "total_tokens": record.total_tokens,
                "latency_ms": response.latency_ms,
                "estimated_cost": estimate_cost(
                    record.input_tokens, record.output_tokens, self.pricing
                ),
                "cache_state": "disabled",
                "error_type": None,
            }
        )
        return response
