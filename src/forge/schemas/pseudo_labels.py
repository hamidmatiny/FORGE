"""Pseudo-labels table schema — one row per fused object, confidence-gated."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion

# "auto_accept": trust_score cleared the high threshold -- usable as a
#   label without human review.
# "needs_review": between the two thresholds -- routed to a human review
#   queue, ranked by review_priority.
# "rejected": below the low threshold -- too unreliable to be worth a
#   reviewer's time; kept in the table (not silently dropped) so the
#   decision is auditable.
Decision = str


class PseudoLabelRecord(BaseModel):
    """A fused object scored and routed by the active-learning / pseudo-labeling policy."""

    model_config = ConfigDict(frozen=True)

    pseudo_label_id: str = Field(description="Unique identifier for this row.")
    fusion_id: str = Field(description="Source fused_objects.fusion_id.")
    scene_id: str = Field(description="Scene this object belongs to.")
    timestamp_us: int = Field(ge=0, description="Sample timestamp in microseconds.")
    fusion_type: str = Field(
        description="Carried through from fused_objects: matched/camera_only/lidar_only."
    )
    class_id: int = Field(ge=0, description="Class index.")
    class_name: str = Field(description="Human-readable class label.")
    bbox_xyxy: list[float] = Field(
        min_length=4, max_length=4, description="Carried through from fused_objects."
    )
    center_xyz: list[float] = Field(
        min_length=3, max_length=3, description="Carried through from fused_objects."
    )
    dimensions_whl: list[float] = Field(
        min_length=3, max_length=3, description="Carried through from fused_objects."
    )
    yaw: float = Field(description="Carried through from fused_objects.")
    trust_score: float = Field(
        ge=0.0, le=1.0, description="Cross-modal-agreement-adjusted confidence, see DECISIONS.md."
    )
    decision: str = Field(description="'auto_accept', 'needs_review', or 'rejected'.")
    review_priority: float = Field(
        ge=0.0,
        description="Binary entropy of trust_score — higher means more valuable to review first.",
    )
    labeler_version: str = Field(description="Labeling run/config identifier.")


class PseudoLabelsTable(BaseTable[PseudoLabelRecord]):
    """Versioned pseudo-labels table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "pseudo_labels"
    record_model: ClassVar[type[PseudoLabelRecord]] = PseudoLabelRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("pseudo_label_id", pa.string(), nullable=False),
                pa.field("fusion_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("fusion_type", pa.string(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("center_xyz", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("dimensions_whl", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("yaw", pa.float32(), nullable=False),
                pa.field("trust_score", pa.float32(), nullable=False),
                pa.field("decision", pa.string(), nullable=False),
                pa.field("review_priority", pa.float32(), nullable=False),
                pa.field("labeler_version", pa.string(), nullable=False),
            ]
        )
