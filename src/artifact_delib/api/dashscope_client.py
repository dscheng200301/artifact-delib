"""DashScope (Alibaba Cloud Model Studio) model client for Qwen VLMs.

Reads API key from DASHSCOPE_API_KEY environment variable.
Uses the OpenAI-compatible /chat/completions endpoint.

Usage:
    client = DashScopeModelClient(model="qwen3.5-flash-2026-02-23")
    response = client.generate(request)
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from artifact_delib.api.schemas import ModelRequest, ModelResponse, TokenUsage

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeModelClient:
    """DashScope (Qwen) multimodal model client.

    Supports image+text input via the OpenAI-compatible API.
    Reads DASHSCOPE_API_KEY from environment; fails closed if missing.
    """

    def __init__(
        self,
        model: str = "qwen3.5-flash-2026-02-23",
        base_url: str = _DASHSCOPE_BASE_URL,
        timeout_seconds: int = 120,
    ) -> None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required for remote execution. "
                "Set it as an environment variable."
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Send a request to DashScope and return a normalized response."""
        messages = self._build_messages(request)
        payload = self._build_payload(request, messages)

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise RuntimeError(
                    "DashScope API authentication failed (401). "
                    "Check your DASHSCOPE_API_KEY."
                ) from exc
            if exc.response.status_code == 429:
                raise RuntimeError(
                    "DashScope API rate limit exceeded (429). "
                    "Reduce concurrency or wait before retrying."
                ) from exc
            raise RuntimeError(
                f"DashScope API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"DashScope API timeout after {self._timeout_seconds}s"
            ) from exc
        except httpx.NetworkError as exc:
            raise ConnectionError(
                f"DashScope API network error: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        body = response.json()
        usage = body.get("usage", {})
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content")
        if content is None:
            finish_reason = choice.get("finish_reason", "unknown")
            raise ValueError(
                f"DashScope returned null content (finish_reason={finish_reason})"
            )

        return ModelResponse(
            request_id=request.request_id,
            content=str(content),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                total_latency_ms=elapsed_ms,
            ),
            latency_ms=elapsed_ms,
            provider="dashscope",
            model=request.model,
        )

    def _build_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        """Build the messages array for the chat completion API."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
        ]

        if request.image_base64:
            # Multimodal: user message with text + image
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": request.user_prompt},
                {"type": "image_url", "image_url": {"url": request.image_base64}},
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": request.user_prompt})

        return messages

    def _build_payload(
        self,
        request: ModelRequest,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.response_schema is not None:
            payload["response_format"] = request.response_schema
        return payload