from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from histodelib.api.budget import BudgetExceeded, BudgetManager
from histodelib.api.cache import ResponseCache
from histodelib.api.call_log import CallLogStore, redact_secrets
from histodelib.api.cost import estimate_cost
from histodelib.api.mock import MockModelClient
from histodelib.api.openai_compatible import OpenAICompatibleClient
from histodelib.api.response_parser import StructuredResponseError, parse_json_object
from histodelib.api.retry import retry_call
from histodelib.api.token_usage import TokenAccounting
from histodelib.schemas import ModelRequest, TokenUsage


def test_mock_client_is_deterministic_and_records_usage() -> None:
    client = MockModelClient(role="llm")
    request = ModelRequest(
        request_id="req-1",
        model="fixture-model",
        system_prompt="Return JSON.",
        user_prompt="caption says 1912 harbor",
    )

    first = client.generate(request)
    second = client.generate(request)

    assert first.content == second.content
    assert first.usage.total_tokens > 0
    assert first.provider == "mock"


def test_budget_refuses_call_after_request_limit() -> None:
    budget = BudgetManager(max_requests=1, max_tokens=100, max_cost=1.0)
    budget.reserve(TokenUsage(input_tokens=2, output_tokens=3), estimated_cost=0.01)

    with pytest.raises(BudgetExceeded):
        budget.reserve(TokenUsage(input_tokens=2, output_tokens=3), estimated_cost=0.01)


def test_file_cache_returns_existing_value_without_recomputing(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    calls = 0

    def producer() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"state": "fresh"}

    first, first_state = cache.get_or_set("safe-key", producer)
    second, second_state = cache.get_or_set("safe-key", producer)

    assert first == second == {"state": "fresh"}
    assert (first_state, second_state, calls) == ("miss", "hit", 1)
    assert json.loads((tmp_path / "safe-key.json").read_text(encoding="utf-8")) == first


def test_response_parser_repairs_fenced_json_and_rejects_non_object() -> None:
    assert parse_json_object('```json\n{"label": "TRUE"}\n```') == {"label": "TRUE"}

    with pytest.raises(StructuredResponseError):
        parse_json_object("[1, 2]")


def test_retry_call_retries_transient_failures_only() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    assert retry_call(operation, max_retries=3) == "ok"
    assert attempts == 3


def test_token_accounting_records_call_summary() -> None:
    accounting = TokenAccounting()
    record = accounting.record("req-1", input_tokens=4, output_tokens=6, latency_ms=12.5)

    assert record.total_tokens == 10
    assert accounting.total_tokens == 10
    assert accounting.total_calls == 1


@respx.mock
def test_openai_compatible_client_normalizes_http_response() -> None:
    route = respx.post("https://example.test/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [{"message": {"content": '{"label":"TRUE"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )
    )
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="secret",
        timeout_seconds=5,
        allow_paid_calls=True,
    )
    request = ModelRequest(
        request_id="req-http",
        model="remote-model",
        system_prompt="system",
        user_prompt="caption",
    )

    response = client.generate(request)

    assert route.called
    assert response.usage.total_tokens == 10
    assert response.provider == "openai_compatible"


def test_call_log_redacts_authorization_and_cost_is_optional(tmp_path) -> None:
    assert "secret" not in redact_secrets("Authorization: Bearer secret")
    store = CallLogStore(tmp_path / "calls.jsonl")
    store.append({"error": "api_key=secret", "request_id": "req-1"})

    assert "secret" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert estimate_cost(1000, 500, {"input_per_million": 1.0, "output_per_million": 2.0}) == 0.002
    assert estimate_cost(1000, 500, None) is None
