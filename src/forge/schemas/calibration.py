"""Calibration table schema — one row per unique calibrated sensor."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class CalibrationRecord(BaseModel):
    """Extrinsic (translation/rotation) and optional intrinsic calibration for a sensor."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(description="Unique calibrated-sensor identifier.")
    sensor_id: str = Field(description="Sensor channel identifier (e.g. CAM_FRONT).")
    translation: list[float] = Field(
        min_length=3, max_length=3, description="Sensor-to-ego translation [x, y, z] in meters."
    )
    rotation: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Sensor-to-ego rotation quaternion [w, x, y, z].",
    )
    camera_intrinsic: list[float] = Field(
        default_factory=list,
        description="Flattened 3x3 camera intrinsic matrix, row-major; empty for non-cameras.",
    )


class CalibrationTable(BaseTable[CalibrationRecord]):
    """Versioned calibration table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "calibration"
    record_model: ClassVar[type[CalibrationRecord]] = CalibrationRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("token", pa.string(), nullable=False),
                pa.field("sensor_id", pa.string(), nullable=False),
                pa.field("translation", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("rotation", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("camera_intrinsic", pa.list_(pa.float64()), nullable=False),
            ]
        )
