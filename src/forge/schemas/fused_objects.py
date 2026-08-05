"""Fused-objects table schema — one row per object after camera/lidar fusion."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion

# "matched": a lidar detection's projected box matched a camera detection.
# "camera_only": a camera detection with no corresponding lidar match
#   (either no synchronized lidar frame, or no lidar detection projected
#   close enough to any of it).
# "lidar_only": a lidar detection with no corresponding camera match
#   (either its projection fell outside the image / behind the camera,
#   or no camera detection matched its projected box).
FusionType = str


class FusedObjectRecord(BaseModel):
    """One object after associating a camera detection with a projected lidar detection."""

    model_config = ConfigDict(frozen=True)

    fusion_id: str = Field(description="Unique identifier for this fused-object row.")
    scene_id: str = Field(description="Scene the fusion happened within.")
    timestamp_us: int = Field(ge=0, description="Synchronized sample timestamp in microseconds.")
    fusion_type: str = Field(description="'matched', 'camera_only', or 'lidar_only'.")
    frame_id_2d: str = Field(description="Camera frame_id, or '' if this row has no 2D side.")
    frame_id_3d: str = Field(description="Lidar frame_id, or '' if this row has no 3D side.")
    detection_id_2d: str = Field(description="detections_2d.detection_id, or '' if none.")
    detection_id_3d: str = Field(description="detections_3d.detection_id, or '' if none.")
    class_id: int = Field(ge=0, description="Class index, preferring the 3D side when both exist.")
    class_name: str = Field(description="Human-readable class label.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence, preferring the 3D side.")
    bbox_xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Best-available 2D footprint: the camera detection's box if present, "
            "else the lidar box projected into the camera plane, else [0,0,0,0]."
        ),
    )
    center_xyz: list[float] = Field(
        min_length=3, max_length=3, description="3D center in meters, ego frame; [0,0,0] if none."
    )
    dimensions_whl: list[float] = Field(
        min_length=3, max_length=3, description="3D size [w,h,l] in meters; [0,0,0] if none."
    )
    yaw: float = Field(description="Heading in radians; 0.0 if no 3D side.")
    fuser_version: str = Field(description="Fusion run/config identifier.")


class FusedObjectsTable(BaseTable[FusedObjectRecord]):
    """Versioned fused-objects table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "fused_objects"
    record_model: ClassVar[type[FusedObjectRecord]] = FusedObjectRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("fusion_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("fusion_type", pa.string(), nullable=False),
                pa.field("frame_id_2d", pa.string(), nullable=False),
                pa.field("frame_id_3d", pa.string(), nullable=False),
                pa.field("detection_id_2d", pa.string(), nullable=False),
                pa.field("detection_id_3d", pa.string(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("score", pa.float32(), nullable=False),
                pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("center_xyz", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("dimensions_whl", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("yaw", pa.float32(), nullable=False),
                pa.field("fuser_version", pa.string(), nullable=False),
            ]
        )
