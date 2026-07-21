"""Model provider wrappers — delegates to histodelib.api infrastructure."""

from __future__ import annotations

from artifact_delib.api.base import ModelClient  # noqa: F401
from artifact_delib.api.schemas import ModelRequest, ModelResponse, TokenUsage  # noqa: F401


def build_artifact_model_client(
    settings_path: str | None = None,
    *,
    use_mock: bool = True,
    role: str = "artifact_vlm",
) -> ModelClient:
    """Build a model client for artifact identification.

    During Phase 1, defaults to mock client so the pipeline runs without an API key.
    """
    if use_mock:
        from artifact_delib.models.mock_artifact import ArtifactMockClient
        return ArtifactMockClient(role=role)
    from pathlib import Path
    from artifact_delib.api.factory import build_remote_client
    from artifact_delib.api.settings import Settings

    settings = Settings(_env_file=settings_path) if settings_path else Settings()
    run_dir = settings.artifact_delib_output_root / "artifact_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return build_remote_client(settings, run_dir)
