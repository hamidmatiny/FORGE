"""Tracks table schema — one row per detection, tagged with its assigned track."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class TrackRecord(BaseModel):
    """A single detection, associated to a track that persists across frames."""

    model_config = ConfigDict(frozen=True)

    track_id: str = Field(description="Identifier shared by every detection in this track.")
    detection_id: str = Field(description="The detections_2d.detection_id this row wraps.")
    frame_id: str = Field(description="FrameRecord.frame_id this detection belongs to.")
    scene_id: str = Field(description="Scene the track lives within (tracks never span scenes).")
    sensor_id: str = Field(description="Sensor channel (e.g. CAM_FRONT).")
    timestamp_us: int = Field(ge=0, description="Frame timestamp in microseconds.")
    class_id: int = Field(ge=0, description="Detection's predicted class index.")
    class_name: str = Field(description="Detection's predicted class label.")
    bbox_xyxy: list[float] = Field(
        min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixel coordinates."
    )
    score: float = Field(ge=0.0, le=1.0, description="Detection confidence score.")
    track_age: int = Field(ge=0, description="Number of consecutive frames this track has matched.")
    tracker_version: str = Field(description="Identifier for the tracking run/config used.")


class TracksTable(BaseTable[TrackRecord]):
    """Versioned tracks table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "tracks"
    record_model: ClassVar[type[TrackRecord]] = TrackRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("track_id", pa.string(), nullable=False),
                pa.field("detection_id", pa.string(), nullable=False),
                pa.field("frame_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("sensor_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("score", pa.float32(), nullable=False),
                pa.field("track_age", pa.int32(), nullable=False),
                pa.field("tracker_version", pa.string(), nullable=False),
            ]
        )
