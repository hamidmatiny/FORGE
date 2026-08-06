"""Curated-dataset table schema — pseudo-labels after near-duplicate resolution."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class CuratedRecord(BaseModel):
    """A pseudo-label after curation: kept as-is, or flagged as a near-duplicate."""

    model_config = ConfigDict(frozen=True)

    pseudo_label_id: str = Field(description="Source pseudo_labels.pseudo_label_id.")
    scene_id: str = Field(description="Scene this object belongs to.")
    timestamp_us: int = Field(ge=0, description="Sample timestamp in microseconds.")
    class_id: int = Field(ge=0, description="Class index.")
    class_name: str = Field(description="Human-readable class label.")
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4, description="Carried through.")
    center_xyz: list[float] = Field(min_length=3, max_length=3, description="Carried through.")
    dimensions_whl: list[float] = Field(min_length=3, max_length=3, description="Carried through.")
    yaw: float = Field(description="Carried through.")
    trust_score: float = Field(ge=0.0, le=1.0, description="Carried through from pseudo_labels.")
    is_duplicate: bool = Field(
        description="True if a higher-trust near-duplicate was kept instead."
    )
    duplicate_of_id: str = Field(
        description="pseudo_label_id of the kept near-duplicate, or '' if this row was kept."
    )
    curation_version: str = Field(description="Curation run/config identifier.")


class CuratedTable(BaseTable[CuratedRecord]):
    """Versioned curated-dataset table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "curated"
    record_model: ClassVar[type[CuratedRecord]] = CuratedRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("pseudo_label_id", pa.string(), nullable=False),
                pa.field("scene_id", pa.string(), nullable=False),
                pa.field("timestamp_us", pa.int64(), nullable=False),
                pa.field("class_id", pa.int32(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
                pa.field("center_xyz", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("dimensions_whl", pa.list_(pa.float64(), 3), nullable=False),
                pa.field("yaw", pa.float32(), nullable=False),
                pa.field("trust_score", pa.float32(), nullable=False),
                pa.field("is_duplicate", pa.bool_(), nullable=False),
                pa.field("duplicate_of_id", pa.string(), nullable=False),
                pa.field("curation_version", pa.string(), nullable=False),
            ]
        )
