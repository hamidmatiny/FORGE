"""Eval-metrics table schema — one row per (class, run) evaluation result."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from forge.schemas.base import BaseTable, SchemaVersion


class EvalMetricRecord(BaseModel):
    """Precision/recall/AP for one class (or 'overall') in one evaluation run."""

    model_config = ConfigDict(frozen=True)

    eval_run_id: str = Field(description="Identifier shared by every row from one evaluation run.")
    class_name: str = Field(description="Category name, or 'overall' for the micro-averaged row.")
    num_gt: int = Field(ge=0, description="Number of ground-truth boxes for this class.")
    num_predictions: int = Field(
        ge=0, description="Number of pseudo-label predictions for this class."
    )
    num_matched: int = Field(
        ge=0, description="Predictions matched to a GT box within the distance threshold."
    )
    precision: float = Field(ge=0.0, le=1.0, description="num_matched / num_predictions.")
    recall: float = Field(ge=0.0, le=1.0, description="num_matched / num_gt.")
    f1: float = Field(ge=0.0, le=1.0, description="Harmonic mean of precision and recall.")
    average_precision: float = Field(
        ge=0.0, le=1.0, description="AP from the precision-recall curve."
    )
    distance_threshold_m: float = Field(
        ge=0.0, description="BEV center-distance match threshold used."
    )
    eval_version: str = Field(description="Evaluation run/config identifier.")


class EvalMetricsTable(BaseTable[EvalMetricRecord]):
    """Versioned eval-metrics table."""

    schema_version: ClassVar[SchemaVersion] = SchemaVersion(major=1, minor=0)
    table_name: ClassVar[str] = "eval_metrics"
    record_model: ClassVar[type[EvalMetricRecord]] = EvalMetricRecord

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        return pa.schema(
            [
                pa.field("eval_run_id", pa.string(), nullable=False),
                pa.field("class_name", pa.string(), nullable=False),
                pa.field("num_gt", pa.int32(), nullable=False),
                pa.field("num_predictions", pa.int32(), nullable=False),
                pa.field("num_matched", pa.int32(), nullable=False),
                pa.field("precision", pa.float32(), nullable=False),
                pa.field("recall", pa.float32(), nullable=False),
                pa.field("f1", pa.float32(), nullable=False),
                pa.field("average_precision", pa.float32(), nullable=False),
                pa.field("distance_threshold_m", pa.float32(), nullable=False),
                pa.field("eval_version", pa.string(), nullable=False),
            ]
        )
