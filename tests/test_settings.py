"""Tests for application settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.settings import ForgeSettings


def test_default_settings() -> None:
    settings = ForgeSettings()
    assert settings.data_lake_root == Path("data/lake")
    assert settings.mlflow_uri == "file:./mlruns"
    assert settings.log_level == "INFO"


def test_env_prefix_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FORGE_DATA_LAKE_ROOT", "/tmp/lake")
    settings = ForgeSettings()
    assert settings.log_level == "DEBUG"
    assert settings.data_lake_root == Path("/tmp/lake")
