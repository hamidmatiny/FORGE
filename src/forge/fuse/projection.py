"""Pinhole camera projection of ego-frame 3D boxes into a camera's image plane.

Standard calibrated-camera geometry, using the calibration extrinsics
(sensor-to-ego translation + rotation) and intrinsics recorded at ingest
time (Phase 1). No learning involved — this is the same math any
camera-lidar fusion pipeline uses to relate the two modalities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from forge.schemas import CalibrationRecord

Point3D = tuple[float, float, float]
BBox2D = tuple[float, float, float, float]


def quaternion_to_rotation_matrix(quaternion: list[float]) -> npt.NDArray[np.float64]:
    """[w, x, y, z] -> 3x3 rotation matrix (sensor-frame axes expressed in ego frame)."""
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ego_to_sensor(point_ego: Point3D, calibration: CalibrationRecord) -> npt.NDArray[np.float64]:
    """Transform a point from the ego frame into the given sensor's frame."""
    rotation = quaternion_to_rotation_matrix(calibration.rotation)
    translation = np.array(calibration.translation, dtype=np.float64)
    point = np.array(point_ego, dtype=np.float64)
    return rotation.T @ (point - translation)


def project_point(
    point_sensor: npt.NDArray[np.float64], camera_intrinsic: list[float]
) -> tuple[float, float] | None:
    """Project a point already in the camera's sensor frame to a pixel coordinate.

    Returns None if the point is behind (or at) the camera — it can't
    project to a valid pixel.
    """
    if point_sensor[2] <= 1e-3:
        return None
    k_matrix = np.array(camera_intrinsic, dtype=np.float64).reshape(3, 3)
    pixel = k_matrix @ point_sensor
    return (float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]))


def box_corners_3d(
    center: list[float], dimensions_whl: list[float], yaw: float
) -> npt.NDArray[np.float64]:
    """The 8 corners of a 3D box (ego frame), given center, [w,h,l], and yaw about z."""
    cx, cy, cz = center
    w, h, length = dimensions_whl
    half_l, half_w, half_h = length / 2.0, w / 2.0, h / 2.0

    local_corners = np.array(
        [
            [half_l, half_w, half_h],
            [half_l, -half_w, half_h],
            [-half_l, -half_w, half_h],
            [-half_l, half_w, half_h],
            [half_l, half_w, -half_h],
            [half_l, -half_w, -half_h],
            [-half_l, -half_w, -half_h],
            [-half_l, half_w, -half_h],
        ],
        dtype=np.float64,
    )

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    rotated = (rotation_z @ local_corners.T).T
    return rotated + np.array([cx, cy, cz], dtype=np.float64)


def project_box_to_bbox(
    center: list[float],
    dimensions_whl: list[float],
    yaw: float,
    calibration: CalibrationRecord,
) -> BBox2D | None:
    """Project a 3D ego-frame box into a camera's image plane as an enclosing 2D bbox.

    Returns None if every corner is behind the camera (nothing to project).
    Corners that ARE in front are still included even if others aren't —
    this deliberately produces a wider, more inclusive box for partially
    visible objects rather than dropping them.
    """
    if not calibration.camera_intrinsic:
        raise ValueError(
            f"Calibration for sensor '{calibration.sensor_id}' has no camera_intrinsic "
            "— it isn't a camera, projection isn't meaningful."
        )

    corners_ego = box_corners_3d(center, dimensions_whl, yaw)
    pixels: list[tuple[float, float]] = []
    for corner in corners_ego:
        point_sensor = ego_to_sensor((corner[0], corner[1], corner[2]), calibration)
        pixel = project_point(point_sensor, calibration.camera_intrinsic)
        if pixel is not None:
            pixels.append(pixel)

    if not pixels:
        return None

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return (min(xs), min(ys), max(xs), max(ys))
