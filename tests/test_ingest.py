"""Tests for nuScenes-format ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.ingest import ingest_nuscenes
from forge.schemas import CalibrationTable, EgoPoseTable, FramesTable

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "nuscenes_mini_synthetic"


def test_ingest_keyframes_only(tmp_path: Path) -> None:
    result = ingest_nuscenes(
        input_dir=FIXTURE_ROOT,
        lake_root=tmp_path,
        version="v1.0-mini",
        split="mini_train",
        key_frames_only=True,
    )
    assert result.frames_written == 5  # one non-keyframe sweep is excluded
    assert result.calibration_written == 2  # CAM_FRONT + LIDAR_TOP, deduped
    assert result.ego_poses_written == 3  # deduped across shared ego-pose tokens
    assert result.scenes_seen == 2


def test_ingest_all_sweeps(tmp_path: Path) -> None:
    result = ingest_nuscenes(
        input_dir=FIXTURE_ROOT,
        lake_root=tmp_path,
        key_frames_only=False,
    )
    assert result.frames_written == 6  # the sweep is now included


def test_ingest_sensor_filter(tmp_path: Path) -> None:
    result = ingest_nuscenes(
        input_dir=FIXTURE_ROOT,
        lake_root=tmp_path,
        sensors=["LIDAR_TOP"],
    )
    assert result.frames_written == 2
    assert result.calibration_written == 1


def test_ingest_writes_readable_parquet(tmp_path: Path) -> None:
    ingest_nuscenes(input_dir=FIXTURE_ROOT, lake_root=tmp_path, split="mini_val")

    frames = FramesTable.read_parquet(str(tmp_path / "frames.parquet"))
    assert all(f.dataset_split == "mini_val" for f in frames)
    assert {f.sensor_id for f in frames} == {"CAM_FRONT", "LIDAR_TOP"}
    assert all(f.data_path for f in frames)

    calibration = CalibrationTable.read_parquet(str(tmp_path / "calibration.parquet"))
    cam_calib = next(c for c in calibration if c.sensor_id == "CAM_FRONT")
    assert len(cam_calib.camera_intrinsic) == 9  # flattened 3x3
    lidar_calib = next(c for c in calibration if c.sensor_id == "LIDAR_TOP")
    assert lidar_calib.camera_intrinsic == []

    ego_poses = EgoPoseTable.read_parquet(str(tmp_path / "ego_pose.parquet"))
    assert len(ego_poses) == 3


def test_ingest_missing_table_raises_clear_error(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    (empty_dir / "v1.0-mini").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="scene.json"):
        ingest_nuscenes(input_dir=empty_dir, lake_root=tmp_path / "lake")


def test_ingest_wrong_version_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_nuscenes(
            input_dir=FIXTURE_ROOT,
            lake_root=tmp_path,
            version="v1.0-trainval",
        )
