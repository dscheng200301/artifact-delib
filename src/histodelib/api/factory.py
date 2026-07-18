"""Construct guarded remote API clients from redacted runtime settings."""

from __future__ import annotations

from pathlib import Path

from histodelib.api.budget import BudgetManager
from histodelib.api.cache import ResponseCache
from histodelib.api.call_log import CallLogStore
from histodelib.api.cost import load_pricing
from histodelib.api.guarded import GuardedModelClient
from histodelib.api.openai_compatible import OpenAICompatibleClient
from histodelib.constants import DEFAULT_MODEL
from histodelib.settings import Settings


def build_remote_client(settings: Settings, run_dir: Path) -> GuardedModelClient:
    """Build a paid-call client only when explicit settings authorize it."""

    if not settings.api_allow_paid_calls:
        raise PermissionError("set API_ALLOW_PAID_CALLS=true before a remote run")
    api_key = settings.vlm_api_key or settings.llm_api_key
    base_url = settings.vlm_base_url or settings.llm_base_url
    if not api_key or not base_url:
        raise ValueError("API key and base URL are required for a remote run")
    if (
        settings.vlm_provider != "openai_compatible"
        and settings.llm_provider != "openai_compatible"
    ):
        raise ValueError("only openai_compatible provider is supported by this factory")

    provider = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=settings.api_timeout_seconds,
        allow_paid_calls=True,
    )
    pricing_path = Path("configs/api/pricing.example.yaml")
    pricing = (
        load_pricing(pricing_path, selected_model(settings))
        if pricing_path.exists()
        else None
    )
    return GuardedModelClient(
        provider,
        cache=ResponseCache(run_dir / "cache"),
        budget=BudgetManager(
            max_requests=settings.api_max_total_requests,
            max_tokens=settings.api_max_total_tokens,
            max_cost=settings.api_max_estimated_cost,
        ),
        call_log=CallLogStore(run_dir / "call_log.jsonl"),
        max_retries=settings.api_max_retries,
        pricing=pricing,
        max_concurrency=settings.api_max_concurrency,
    )


def selected_model(settings: Settings) -> str:
    """Return the configured vision/text model for a unified-model run."""

    return settings.vlm_model or settings.llm_model or DEFAULT_MODEL
