"""Ground-truth table schema — nuScenes human annotations, evaluation-only.

Per the dataset notice in README.md, these are used only to *score* the
pipeline's auto-labels — never as an input anywhere upstream in
ingest/detect2d/detect3d/track/fuse/label.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class GroundTruthRecord(BaseModel):
    """A single human-annotated 3D box from nuScenes, for evaluation only."""

    model_config = ConfigDict(frozen=True)

    annotation_id: str = Field(description="nuScenes sample_annotation token.")
    scene_id: str = Field(description="Scene this annotation belongs to.")
    timestamp_us: int = Field(ge=0, description="Sample timestamp in microseconds.")
    category_name: str = Field(description="nuScenes category name (flattened, see DECISIONS.md).")
    center_xyz: list[float] = Field(
        min_length=3, max_length=3, description="Box center in meters, global/ego frame."
    )
    dimensions_whl: list[float] = Field(
        min_length=3,
        max_length=3,
        description="Box size [width, height, length], FORGE's order (matches detections_3d).",
    )
    yaw: float = Field(description="Heading in radians, extracted from the annotation quaternion.")
    num_lidar_pts: int = Field(ge=0, description="Lidar points inside this box, per nuScenes.")


class GroundTruthTable(BaseTable[GroundTruthRecord]):
    """Versioned ground-truth table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "ground_truth"
    record_model: ClassVar[type[GroundTruthRecord]] = GroundTruthRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("annotation_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("category_name", pa.string(), nullable=False),
                pa.field("center_xyz", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("dimensions_whl", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("yaw", pa.float32(), nullable=False),
                pa.field("num_lidar_pts", pa.int32(), nullable=False),
            ]
        )
