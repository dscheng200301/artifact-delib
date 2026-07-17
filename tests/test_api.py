from __future__ import annotations

import json

import pytest

from histodelib.api.budget import BudgetExceeded, BudgetManager
from histodelib.api.cache import ResponseCache
from histodelib.api.mock import MockModelClient
from histodelib.api.response_parser import StructuredResponseError, parse_json_object
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
