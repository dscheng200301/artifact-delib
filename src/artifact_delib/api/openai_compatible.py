"""Minimal OpenAI-compatible HTTP transport, disabled unless explicitly authorized."""

from __future__ import annotations

import time
from typing import Any

import httpx

from artifact_delib.api.schemas import ModelRequest, ModelResponse, TokenUsage


class OpenAICompatibleClient:
    """Call `/chat/completions` only when constructed with explicit authorization."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        allow_paid_calls: bool,
    ) -> None:
        if not allow_paid_calls:
            raise PermissionError("paid API calls are disabled; set API_ALLOW_PAID_CALLS=true")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def generate(self, request: ModelRequest) -> ModelResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        if request.image_base64 is not None:
            messages[1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.user_prompt},
                    {"type": "image_url", "image_url": {"url": request.image_base64}},
                ],
            }
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.response_schema is not None:
            payload["response_format"] = request.response_schema
            # DashScope JSON mode is incompatible with Qwen hybrid thinking mode.
            payload["enable_thinking"] = False
        started = time.perf_counter()
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content")
        if content is None:
            raw = choice.get("finish_reason", "unknown")
            raise ValueError(f"API returned null content (finish_reason={raw})")
        return ModelResponse(
            request_id=request.request_id,
            content=str(content),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            provider="openai_compatible",
            model=request.model,
        )
