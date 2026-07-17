"""Provider-independent model client contracts."""

from __future__ import annotations

from typing import Protocol

from histodelib.schemas import ModelRequest, ModelResponse


class ModelClient(Protocol):
    """A synchronous client that normalizes provider responses."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one structured completion for a normalized request."""
