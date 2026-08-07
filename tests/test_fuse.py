"""Tests for forge.fuse: pinhole projection and camera/lidar fusion orchestration.

Skipped entirely when the [fuse] extra (numpy/scipy) isn't installed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from forge.fuse.projection import (  # noqa: E402
    box_corners_3d,
    ego_to_sensor,
    project_box_to_bbox,
    project_point,
    quaternion_to_rotation_matrix,
)
from forge.fuse.run import run_fusion  # noqa: E402
from forge.schemas import (  # noqa: E402
    CalibrationRecord,
    Detection2DRecord,
    Detection3DRecord,
    FrameRecord,
)

IDENTITY_CAM = CalibrationRecord(
    token="c1",
    sensor_id="CAM_FRONT",
    translation=[0.0, 0.0, 0.0],
    rotation=[1.0, 0.0, 0.0, 0.0],
    camera_intrinsic=[100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
)


def _frame(frame_id: str, scene_id: str, timestamp_us: int, sensor_id: str) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        sensor_id=sensor_id,
        dataset_split="train",
        data_path="x",
        ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


# --- Projection math -------------------------------------------------------


def test_identity_quaternion_gives_identity_rotation() -> None:
    import numpy as np

    rotation = quaternion_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)


def test_ego_to_sensor_identity_calibration_is_passthrough() -> None:
    point = ego_to_sensor((1.0, 2.0, 3.0), IDENTITY_CAM)
    assert list(point) == pytest.approx([1.0, 2.0, 3.0])


def test_project_point_behind_camera_returns_none() -> None:
    import numpy as np

    assert project_point(np.array([0.0, 0.0, -1.0]), IDENTITY_CAM.camera_intrinsic) is None


def test_project_point_at_principal_axis_hits_principal_point() -> None:
    import numpy as np

    pixel = project_point(np.array([0.0, 0.0, 10.0]), IDENTITY_CAM.camera_intrinsic)
    assert pixel is not None
    assert pixel == pytest.approx((50.0, 50.0))


def test_box_corners_3d_count_and_centering() -> None:
    corners = box_corners_3d(center=[0.0, 0.0, 0.0], dimensions_whl=[2.0, 2.0, 2.0], yaw=0.0)
    assert corners.shape == (8, 3)
    assert corners.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_project_box_to_bbox_small_box_ahead_is_near_principal_point() -> None:
    bbox = project_box_to_bbox(
        center=[0.0, 0.0, 10.0], dimensions_whl=[0.1, 0.1, 0.1], yaw=0.0, calibration=IDENTITY_CAM
    )
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert 45 < x1 < 50 < x2 < 55
    assert 45 < y1 < 50 < y2 < 55


def test_project_box_to_bbox_fully_behind_camera_is_none() -> None:
    bbox = project_box_to_bbox(
        center=[0.0, 0.0, -10.0], dimensions_whl=[0.1, 0.1, 0.1], yaw=0.0, calibration=IDENTITY_CAM
    )
    assert bbox is None


def test_project_box_to_bbox_rejects_non_camera_calibration() -> None:
    lidar_calib = CalibrationRecord(
        token="l1",
        sensor_id="LIDAR_TOP",
        translation=[0.0, 0.0, 0.0],
        rotation=[1.0, 0.0, 0.0, 0.0],
        camera_intrinsic=[],
    )
    with pytest.raises(ValueError, match="no camera_intrinsic"):
        project_box_to_bbox([0, 0, 10], [1, 1, 1], 0.0, lidar_calib)


# --- run_fusion orchestration -----------------------------------------------


def test_run_fusion_matches_when_projection_overlaps_camera_box() -> None:
    frames = [
        _frame("cam0", "scene-a", 0, "CAM_FRONT"),
        _frame("lidar0", "scene-a", 0, "LIDAR_TOP"),
    ]
    det_3d = Detection3DRecord(
        detection_id="3d-0",
        frame_id="lidar0",
        class_id=1,
        class_name="vehicle",
        score=0.8,
        center_xyz=[0.0, 0.0, 10.0],
        dimensions_whl=[0.5, 0.5, 0.5],
        yaw=0.0,
        model_version="t",
    )
    det_2d = Detection2DRecord(
        detection_id="2d-0",
        frame_id="cam0",
        class_id=1,
        class_name="vehicle",
        score=0.9,
        bbox_xyxy=[46.0, 46.0, 54.0, 54.0],  # closely brackets the ~(47.4-52.6) projection
        model_version="t",
    )
    fused = run_fusion([det_2d], [det_3d], frames, [IDENTITY_CAM], iou_threshold=0.1)
    assert len(fused) == 1
    assert fused[0].fusion_type == "matched"
    assert fused[0].detection_id_2d == "2d-0"
    assert fused[0].detection_id_3d == "3d-0"


def test_run_fusion_unmatched_camera_detection_is_camera_only() -> None:
    frames = [
        _frame("cam0", "scene-a", 0, "CAM_FRONT"),
        _frame("lidar0", "scene-a", 0, "LIDAR_TOP"),
    ]
    det_2d = Detection2DRecord(
        detection_id="2d-0",
        frame_id="cam0",
        class_id=2,
        class_name="pedestrian",
        score=0.7,
        bbox_xyxy=[500.0, 500.0, 550.0, 550.0],  # far from any projected lidar box
        model_version="t",
    )
    fused = run_fusion([det_2d], [], frames, [IDENTITY_CAM])
    assert len(fused) == 1
    assert fused[0].fusion_type == "camera_only"
    assert fused[0].bbox_xyxy == det_2d.bbox_xyxy


def test_run_fusion_behind_camera_lidar_detection_is_lidar_only_no_bbox() -> None:
    frames = [
        _frame("cam0", "scene-a", 0, "CAM_FRONT"),
        _frame("lidar0", "scene-a", 0, "LIDAR_TOP"),
    ]
    det_3d = Detection3DRecord(
        detection_id="3d-0",
        frame_id="lidar0",
        class_id=1,
        class_name="vehicle",
        score=0.5,
        center_xyz=[0.0, 0.0, -10.0],  # behind the camera
        dimensions_whl=[0.5, 0.5, 0.5],
        yaw=0.0,
        model_version="t",
    )
    fused = run_fusion([], [det_3d], frames, [IDENTITY_CAM])
    assert len(fused) == 1
    assert fused[0].fusion_type == "lidar_only"
    assert fused[0].bbox_xyxy == [0.0, 0.0, 0.0, 0.0]


def test_run_fusion_no_synchronized_lidar_frame_is_camera_only() -> None:
    frames = [_frame("cam0", "scene-a", 0, "CAM_FRONT")]  # no lidar frame at all
    det_2d = Detection2DRecord(
        detection_id="2d-0",
        frame_id="cam0",
        class_id=1,
        class_name="vehicle",
        score=0.9,
        bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
        model_version="t",
    )
    fused = run_fusion([det_2d], [], frames, [IDENTITY_CAM])
    assert len(fused) == 1
    assert fused[0].fusion_type == "camera_only"


def test_run_fusion_missing_calibration_is_camera_only() -> None:
    frames = [
        _frame("cam0", "scene-a", 0, "CAM_FRONT"),
        _frame("lidar0", "scene-a", 0, "LIDAR_TOP"),
    ]
    det_2d = Detection2DRecord(
        detection_id="2d-0",
        frame_id="cam0",
        class_id=1,
        class_name="vehicle",
        score=0.9,
        bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
        model_version="t",
    )
    fused = run_fusion([det_2d], [], frames, calibration=[])  # no calibration at all
    assert len(fused) == 1
    assert fused[0].fusion_type == "camera_only"


def test_run_fusion_distributed_true_produces_same_results_as_sequential() -> None:
    """Ray's API mocked as a real-executing passthrough (see test_detect3d.py's identical

    pattern) -- confirms run_fusion wires the shared lookup dicts through
    shared_args correctly, not just that the right functions get called.
    """
    from unittest.mock import patch

    frames = [
        _frame("cam0", "scene-a", 0, "CAM_FRONT"),
        _frame("cam1", "scene-a", 100, "CAM_FRONT"),
    ]
    det_2d_0 = Detection2DRecord(
        detection_id="2d-0",
        frame_id="cam0",
        class_id=1,
        class_name="vehicle",
        score=0.9,
        bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
        model_version="t",
    )
    det_2d_1 = Detection2DRecord(
        detection_id="2d-1",
        frame_id="cam1",
        class_id=1,
        class_name="vehicle",
        score=0.8,
        bbox_xyxy=[5.0, 5.0, 15.0, 15.0],
        model_version="t",
    )

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_ray.put.side_effect = lambda arg: arg
        mock_ray.remote.side_effect = lambda fn: _FakeRemote(fn)
        mock_ray.get.side_effect = lambda futures: [f() for f in futures]

        fused = run_fusion([det_2d_0, det_2d_1], [], frames, calibration=[], distributed=True)

    assert len(fused) == 2
    assert all(f.fusion_type == "camera_only" for f in fused)


class _FakeRemote:
    """Stand-in for a `ray.remote`-wrapped function: `.remote(*a)` returns a thunk."""

    def __init__(self, fn: object) -> None:
        self._fn = fn

    def remote(self, *args: object) -> object:
        return lambda: self._fn(*args)  # type: ignore[operator]
