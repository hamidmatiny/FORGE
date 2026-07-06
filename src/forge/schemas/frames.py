"""Frames table schema — one row per sensor sample timestamp."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class FrameRecord(BaseModel):
    """A single indexed frame in the FORGE data lake."""

    model_config = ConfigDict(frozen=True)

    frame_id: str = Field(description="Unique frame identifier (UUID or scene-token).")
    scene_id: str = Field(description="Scene or log segment identifier.")
    timestamp_us: int = Field(ge=0, description="Sample timestamp in microseconds.")
    sensor_id: str = Field(description="Sensor channel identifier (e.g. CAM_FRONT).")
    dataset_split: str = Field(
        default="unknown",
        description="Dataset split label (train/val/test/unknown).",
    )
    ingested_at: datetime = Field(description="UTC timestamp when the row was written.")


class FramesTable(BaseTable[FrameRecord]):
    """Versioned frames table for Phase 0."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "frames"
    record_model: ClassVar[type[FrameRecord]] = FrameRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("frame_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("sensor_id", pa.string(), nullable=False),
                pa.field("dataset_split", pa.string(), nullable=False),
                pa.field(
                    "ingested_at",
                    pa.timestamp("us", tz="UTC"),
                    nullable=False,
                ),
            ]
        )
