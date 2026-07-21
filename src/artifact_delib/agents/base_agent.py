"""Agent base class with shared model invocation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_delib.api.base import ModelClient
from artifact_delib.api.schemas import ModelRequest, TokenUsage


class ArtifactAgent:
    """Base class for artifact identification agents."""

    def __init__(
        self,
        client: ModelClient,
        prompt_name: str,
        model_name: str = "default",
    ) -> None:
        self.client = client
        self.prompt_name = prompt_name
        self.model_name = model_name

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        *,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, TokenUsage]:
        """Unified model call with optional image."""
        image_base64 = None
        if image_path is not None and image_path.exists():
            import base64
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            image_base64 = f"data:image/png;base64,{encoded}"

        from uuid import uuid4
        response = self.client.generate(
            ModelRequest(
                request_id=str(uuid4()),
                model=self.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_base64=image_base64,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                prompt_name=self.prompt_name,
                prompt_version="v1",
            )
        )
        return response.content, response.usage
