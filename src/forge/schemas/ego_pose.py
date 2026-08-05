"""Ego-pose table schema — one row per unique vehicle pose."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class EgoPoseRecord(BaseModel):
    """Global vehicle pose at a given timestamp."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(description="Unique ego-pose identifier.")
    timestamp_us: int = Field(ge=0, description="Pose timestamp in microseconds.")
    translation: list[float] = Field(
        min_length=3, max_length=3, description="Global-frame translation [x, y, z] in meters."
    )
    rotation: list[float] = Field(
        min_length=4, max_length=4, description="Global-frame rotation quaternion [w, x, y, z]."
    )


class EgoPoseTable(BaseTable[EgoPoseRecord]):
    """Versioned ego-pose table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "ego_pose"
    record_model: ClassVar[type[EgoPoseRecord]] = EgoPoseRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("token", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("translation", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("rotation", pa.list_(pa.float64(), 4), nullable=False),
            ]
        )
