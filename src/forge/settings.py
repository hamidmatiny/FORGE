"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    """Global FORGE configuration with ``FORGE_`` environment prefix."""

    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_lake_root: Path = Field(
        default=Path("data/lake"),
        description="Root directory for versioned Parquet data lake tables.",
    )
    mlflow_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking URI (wired in a later phase).",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for structlog (DEBUG, INFO, WARNING, ERROR).",
    )


def get_settings() -> ForgeSettings:
    """Return cached settings instance."""
    return ForgeSettings()
