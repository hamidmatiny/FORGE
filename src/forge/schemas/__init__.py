"""Versioned Parquet schema machinery for the FORGE data lake."""

from forge.schemas.base import BaseTable, SchemaVersion
from forge.schemas.calibration import CalibrationRecord, CalibrationTable
from forge.schemas.detections_2d import Detection2DRecord, Detections2DTable
from forge.schemas.detections_3d import Detection3DRecord, Detections3DTable
from forge.schemas.ego_pose import EgoPoseRecord, EgoPoseTable
from forge.schemas.frames import FrameRecord, FramesTable
from forge.schemas.tracks import TrackRecord, TracksTable

__all__ = [
    "BaseTable",
    "CalibrationRecord",
    "CalibrationTable",
    "Detection2DRecord",
    "Detection3DRecord",
    "Detections2DTable",
    "Detections3DTable",
    "EgoPoseRecord",
    "EgoPoseTable",
    "FrameRecord",
    "FramesTable",
    "SchemaVersion",
    "TrackRecord",
    "TracksTable",
]
