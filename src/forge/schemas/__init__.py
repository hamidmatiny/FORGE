"""Versioned Parquet schema machinery for the FORGE data lake."""

from forge.schemas.base import BaseTable, SchemaVersion
from forge.schemas.calibration import CalibrationRecord, CalibrationTable
from forge.schemas.detections_2d import Detection2DRecord, Detections2DTable
from forge.schemas.ego_pose import EgoPoseRecord, EgoPoseTable
from forge.schemas.frames import FrameRecord, FramesTable

__all__ = [
    "BaseTable",
    "CalibrationRecord",
    "CalibrationTable",
    "Detection2DRecord",
    "Detections2DTable",
    "EgoPoseRecord",
    "EgoPoseTable",
    "FrameRecord",
    "FramesTable",
    "SchemaVersion",
]
