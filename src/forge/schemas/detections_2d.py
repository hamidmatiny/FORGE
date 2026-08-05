"""2D detections table schema — one row per predicted bounding box."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class Detection2DRecord(BaseModel):
    """A single predicted 2D bounding box on a camera frame."""

    model_config = ConfigDict(frozen=True)

    detection_id: str = Field(description="Unique detection identifier.")
    frame_id: str = Field(description="FrameRecord.frame_id this detection belongs to.")
    class_id: int = Field(ge=0, description="Predicted class index (0 = background/unused).")
    class_name: str = Field(description="Human-readable class label.")
    score: float = Field(ge=0.0, le=1.0, description="Model confidence score.")
    bbox_xyxy: list[float] = Field(
        min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixel coordinates."
    )
    model_version: str = Field(description="Identifier for the model/checkpoint that produced this")


class Detections2DTable(BaseTable[Detection2DRecord]):
    """Versioned 2D detections table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "detections_2d"
    record_model: ClassVar[type[Detection2DRecord]] = Detection2DRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("detection_id", pa.string(), nullable=False),
                pa.field("frame_id", pa.string(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("score", pa.float32(), nullable=False),
                pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("model_version", pa.string(), nullable=False),
            ]
        )
