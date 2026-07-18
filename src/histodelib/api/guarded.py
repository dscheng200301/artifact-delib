"""Guarded provider client with local cache, budget, retry, and audit integration."""

from __future__ import annotations

import hashlib
import json
import threading

from histodelib.api.base import ModelClient
from histodelib.api.budget import BudgetManager
from histodelib.api.cache import ResponseCache
from histodelib.api.call_log import CallLogStore
from histodelib.api.cost import estimate_cost
from histodelib.api.retry import retry_call
from histodelib.api.token_usage import TokenAccounting
from histodelib.schemas import ModelRequest, ModelResponse, TokenUsage


class GuardedModelClient:
    """Apply fail-closed runtime safeguards around any provider-neutral client."""

    def __init__(
        self,
        client: ModelClient,
        cache: ResponseCache,
        budget: BudgetManager,
        call_log: CallLogStore,
        max_retries: int = 0,
        pricing: dict[str, object] | None = None,
        max_concurrency: int = 1,
    ) -> None:
        self.client = client
        self.cache = cache
        self.budget = budget
        self.call_log = call_log
        self.max_retries = max(1, max_retries)
        self.pricing = pricing
        self.accounting = TokenAccounting()
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._concurrency = threading.BoundedSemaphore(max_concurrency)

    def generate(self, request: ModelRequest) -> ModelResponse:
        key = self._cache_key(request)
        cached_payload = self._read_cache(key)
        if cached_payload is not None:
            response = ModelResponse.model_validate(cached_payload)
            self._record(response, cache_state="hit")
            return response

        estimate = self._estimate(request)
        estimated_cost = estimate_cost(
            estimate.input_tokens, estimate.output_tokens, self.pricing
        )
        try:
            # Reserve a conservative upper bound before the provider call.
            self.budget.reserve(estimate, estimated_cost)
            with self._concurrency:
                response = retry_call(
                    lambda: self.client.generate(request), max_retries=self.max_retries
                )
        except Exception as exc:
            status = "BUDGET_EXCEEDED" if exc.__class__.__name__ == "BudgetExceeded" else None
            self.call_log.append(
                {
                    "request_id": request.request_id,
                    "provider": "unknown",
                    "model": request.model,
                    "cache_state": "miss",
                    "error_type": type(exc).__name__,
                    "status": status,
                    "latency_ms": 0.0,
                }
            )
            raise

        self._write_cache(key, response)
        self._record(response, cache_state="miss")
        return response

    @staticmethod
    def _estimate(request: ModelRequest) -> TokenUsage:
        prompt_tokens = max(
            1,
            len((request.system_prompt + " " + request.user_prompt).split()),
        )
        return TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=request.max_output_tokens,
        )

    @staticmethod
    def _cache_key(request: ModelRequest) -> str:
        image_hash = ""
        if request.image_base64:
            image_hash = hashlib.sha256(request.image_base64.encode("ascii")).hexdigest()
        identity = {
            "model": request.model,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "image_sha256": image_hash,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "response_schema": request.response_schema,
        }
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cache(self, key: str) -> dict[str, object] | None:
        target = self.cache.root / f"{key}.json"
        if not target.exists():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, key: str, response: ModelResponse) -> None:
        # Use the cache's lock and validation path without invoking the provider twice.
        self.cache.get_or_set(key, lambda: response.model_dump(mode="json"))

    def _record(self, response: ModelResponse, cache_state: str) -> None:
        record = self.accounting.record(
            response.request_id,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.latency_ms,
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
                "cache_state": cache_state,
                "error_type": None,
                "status": "COMPLETED",
            }
        )
