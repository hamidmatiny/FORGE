"""3D detections table schema — one row per predicted 3D bounding box."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class Detection3DRecord(BaseModel):
    """A single predicted 3D bounding box in the lidar/ego frame."""

    model_config = ConfigDict(frozen=True)

    detection_id: str = Field(description="Unique detection identifier.")
    frame_id: str = Field(description="FrameRecord.frame_id this detection belongs to.")
    class_id: int = Field(ge=0, description="Predicted class index.")
    class_name: str = Field(description="Human-readable class label.")
    score: float = Field(ge=0.0, le=1.0, description="Model confidence score.")
    center_xyz: list[float] = Field(
        min_length=3, max_length=3, description="Box center [x, y, z] in meters, ego frame."
    )
    dimensions_whl: list[float] = Field(
        min_length=3, max_length=3, description="Box size [width, height, length] in meters."
    )
    yaw: float = Field(description="Heading angle in radians, around the vertical (z) axis.")
    model_version: str = Field(description="Identifier for the model/checkpoint used")


class Detections3DTable(BaseTable[Detection3DRecord]):
    """Versioned 3D detections table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "detections_3d"
    record_model: ClassVar[type[Detection3DRecord]] = Detection3DRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("detection_id", pa.string(), nullable=False),
                pa.field("frame_id", pa.string(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("score", pa.float32(), nullable=False),
                pa.field("center_xyz", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("dimensions_whl", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("yaw", pa.float32(), nullable=False),
                pa.field("model_version", pa.string(), nullable=False),
            ]
        )
