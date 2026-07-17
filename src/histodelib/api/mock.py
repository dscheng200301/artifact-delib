"""Deterministic local test double; it never invokes a remote model."""

from __future__ import annotations

import json

from histodelib.schemas import ModelRequest, ModelResponse, TokenUsage


class MockModelClient:
    """Return stable structured evidence from a request for fixture tests."""

    def __init__(self, role: str) -> None:
        self.role = role

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a stable result that preserves the normalized response shape."""

        text = f"{request.system_prompt} {request.user_prompt}".lower()
        if "1913" in text:
            label = "MISCAPTIONED"
        elif "mountain" in text:
            label = "OUT_OF_CONTEXT"
        else:
            label = "TRUE"
        content = json.dumps(
            {"label": label, "role": self.role, "evidence": "deterministic synthetic response"},
            sort_keys=True,
        )
        usage = TokenUsage(input_tokens=max(1, len(text.split())), output_tokens=8)
        return ModelResponse(
            request_id=request.request_id,
            content=content,
            usage=usage,
            latency_ms=0.0,
            provider="mock",
            model=request.model,
        )
