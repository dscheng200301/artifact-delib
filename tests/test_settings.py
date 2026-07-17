from __future__ import annotations

from histodelib.settings import Settings


def test_settings_fail_closed_for_paid_api_calls(tmp_path) -> None:
    settings = Settings(
        histodelib_data_root=tmp_path / "data",
        histodelib_output_root=tmp_path / "outputs",
        histodelib_cache_root=tmp_path / "cache",
    )

    assert settings.api_allow_paid_calls is False
    assert settings.api_max_total_requests == 20
    assert settings.redacted().llm_api_key is None


def test_settings_redacts_configured_api_keys(tmp_path) -> None:
    settings = Settings(
        llm_api_key="secret-value",
        vlm_api_key="another-secret",
        histodelib_data_root=tmp_path / "data",
        histodelib_output_root=tmp_path / "outputs",
        histodelib_cache_root=tmp_path / "cache",
    )

    assert settings.redacted().llm_api_key == "***REDACTED***"
    assert settings.redacted().vlm_api_key == "***REDACTED***"
