"""nuScenes-devkit-format ingestion.

Reads the standard nuScenes JSON table layout (``scene``/``sample``/
``sample_data``/``sensor``/``calibrated_sensor``/``ego_pose`` under
``<input_dir>/<version>/``) and writes FORGE's versioned Parquet lake
tables: ``frames``, ``calibration``, ``ego_pose``.

Only key-frame ``sample_data`` rows are ingested by default in Phase 1;
non-keyframe sweeps are a known gap — see KNOWN_GAPS.md.

nuScenes-mini is never downloaded or committed here: it is licensed for
non-commercial use only and must be obtained separately by the caller.
Tests run against a small synthetic fixture that mirrors the same JSON
table layout (see tests/fixtures/nuscenes_mini_synthetic/).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from forge.schemas import (
    CalibrationRecord,
    CalibrationTable,
    EgoPoseRecord,
    EgoPoseTable,
    FrameRecord,
    FramesTable,
)


class _RawScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    name: str


class _RawSample(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    scene_token: str
    timestamp: int


class _RawSensor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    channel: str


class _RawCalibratedSensor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    sensor_token: str
    translation: list[float]
    rotation: list[float]
    camera_intrinsic: list[list[float]] = []


class _RawEgoPose(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    timestamp: int
    translation: list[float]
    rotation: list[float]


class _RawSampleData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    sample_token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    filename: str
    timestamp: int
    is_key_frame: bool


@dataclass(frozen=True)
class IngestResult:
    """Summary of a completed ingest run."""

    frames_written: int
    calibration_written: int
    ego_poses_written: int
    scenes_seen: int
    output_dir: Path


def _load_table(root: Path, name: str) -> list[dict[str, object]]:
    path = root / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected nuScenes table '{name}.json' under {root}. Confirm "
            "--input-dir points at a directory containing '<version>/*.json' "
            "(the standard nuScenes-devkit layout)."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array.")
    return data


def ingest_nuscenes(
    input_dir: Path,
    lake_root: Path,
    version: str = "v1.0-mini",
    split: str = "mini_train",
    key_frames_only: bool = True,
    sensors: list[str] | None = None,
) -> IngestResult:
    """Transform a nuScenes-devkit-format dataset into the FORGE Parquet lake.

    Args:
        input_dir: Dataset root; must contain ``<version>/*.json``.
        lake_root: Directory the FORGE Parquet lake tables are written to.
        version: nuScenes version directory name (e.g. ``v1.0-mini``).
        split: Dataset split label written into every frame row.
        key_frames_only: If True (Phase 1 default), skip non-keyframe sweeps.
        sensors: If given, only ingest these sensor channels.

    Returns:
        Summary counts of what was written.

    Raises:
        FileNotFoundError: If a required nuScenes JSON table is missing.
    """
    table_root = input_dir / version
    scenes = {s["token"]: _RawScene.model_validate(s) for s in _load_table(table_root, "scene")}
    samples = {s["token"]: _RawSample.model_validate(s) for s in _load_table(table_root, "sample")}
    sensors_by_token = {
        s["token"]: _RawSensor.model_validate(s) for s in _load_table(table_root, "sensor")
    }
    calibrated = {
        c["token"]: _RawCalibratedSensor.model_validate(c)
        for c in _load_table(table_root, "calibrated_sensor")
    }
    ego_poses = {
        e["token"]: _RawEgoPose.model_validate(e) for e in _load_table(table_root, "ego_pose")
    }
    sample_data = [
        _RawSampleData.model_validate(sd) for sd in _load_table(table_root, "sample_data")
    ]

    now = datetime.now(UTC)
    frame_records: list[FrameRecord] = []
    calibration_by_token: dict[str, CalibrationRecord] = {}
    ego_pose_by_token: dict[str, EgoPoseRecord] = {}

    for sd in sample_data:
        if key_frames_only and not sd.is_key_frame:
            continue

        cs = calibrated[sd.calibrated_sensor_token]
        channel = sensors_by_token[cs.sensor_token].channel
        if sensors is not None and channel not in sensors:
            continue

        sample = samples[sd.sample_token]
        scene_name = scenes[sample.scene_token].name

        frame_records.append(
            FrameRecord(
                frame_id=sd.token,
                scene_id=scene_name,
                timestamp_us=sd.timestamp,
                sensor_id=channel,
                dataset_split=split,
                data_path=sd.filename,
                ingested_at=now,
            )
        )

        if cs.token not in calibration_by_token:
            flat_intrinsic = [value for row in cs.camera_intrinsic for value in row]
            calibration_by_token[cs.token] = CalibrationRecord(
                token=cs.token,
                sensor_id=channel,
                translation=cs.translation,
                rotation=cs.rotation,
                camera_intrinsic=flat_intrinsic,
            )

        if sd.ego_pose_token not in ego_pose_by_token:
            ep = ego_poses[sd.ego_pose_token]
            ego_pose_by_token[sd.ego_pose_token] = EgoPoseRecord(
                token=ep.token,
                timestamp_us=ep.timestamp,
                translation=ep.translation,
                rotation=ep.rotation,
            )

    lake_root.mkdir(parents=True, exist_ok=True)
    FramesTable.write_parquet(frame_records, str(lake_root / "frames.parquet"))
    CalibrationTable.write_parquet(
        list(calibration_by_token.values()), str(lake_root / "calibration.parquet")
    )
    EgoPoseTable.write_parquet(
        list(ego_pose_by_token.values()), str(lake_root / "ego_pose.parquet")
    )

    return IngestResult(
        frames_written=len(frame_records),
        calibration_written=len(calibration_by_token),
        ego_poses_written=len(ego_pose_by_token),
        scenes_seen=len(scenes),
        output_dir=lake_root,
    )
