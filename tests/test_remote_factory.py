from __future__ import annotations

import pytest

from histodelib.api.factory import build_remote_client
from histodelib.settings import Settings


def test_remote_factory_requires_key_and_base_url(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key=None,
        llm_base_url=None,
        vlm_api_key=None,
        vlm_base_url=None,
        api_allow_paid_calls=True,
    )

    with pytest.raises(ValueError, match="API key and base URL"):
        build_remote_client(settings, tmp_path / "run")
