"""Versioned Parquet schema machinery for the FORGE data lake."""

from forge.schemas.base import BaseTable, SchemaVersion
from forge.schemas.frames import FrameRecord, FramesTable

__all__ = [
    "BaseTable",
    "FrameRecord",
    "FramesTable",
    "SchemaVersion",
]
