"""Fuse camera and lidar detections via calibrated projection + IoU association."""

from __future__ import annotations

import uuid
from collections import defaultdict

from forge.distributed import run_distributed_map
from forge.fuse.projection import project_box_to_bbox
from forge.schemas import (
    CalibrationRecord,
    Detection2DRecord,
    Detection3DRecord,
    FrameRecord,
    FusedObjectRecord,
)
from forge.track.association import associate

FUSER_VERSION = "geometric-projection-v1"


def _fused_from_match(
    scene_id: str,
    timestamp_us: int,
    det_2d: Detection2DRecord,
    det_3d: Detection3DRecord,
) -> FusedObjectRecord:
    # Prefer the 3D side's class/score when both exist -- it's the side that
    # drove the association geometrically, but this is a judgment call, not
    # a law of nature (documented in DECISIONS.md).
    return FusedObjectRecord(
        fusion_id=str(uuid.uuid4()),
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        fusion_type="matched",
        frame_id_2d=det_2d.frame_id,
        frame_id_3d=det_3d.frame_id,
        detection_id_2d=det_2d.detection_id,
        detection_id_3d=det_3d.detection_id,
        class_id=det_3d.class_id,
        class_name=det_3d.class_name,
        score=det_3d.score,
        bbox_xyxy=det_2d.bbox_xyxy,
        center_xyz=det_3d.center_xyz,
        dimensions_whl=det_3d.dimensions_whl,
        yaw=det_3d.yaw,
        fuser_version=FUSER_VERSION,
    )


def _fused_camera_only(
    scene_id: str, timestamp_us: int, det_2d: Detection2DRecord
) -> FusedObjectRecord:
    return FusedObjectRecord(
        fusion_id=str(uuid.uuid4()),
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        fusion_type="camera_only",
        frame_id_2d=det_2d.frame_id,
        frame_id_3d="",
        detection_id_2d=det_2d.detection_id,
        detection_id_3d="",
        class_id=det_2d.class_id,
        class_name=det_2d.class_name,
        score=det_2d.score,
        bbox_xyxy=det_2d.bbox_xyxy,
        center_xyz=[0.0, 0.0, 0.0],
        dimensions_whl=[0.0, 0.0, 0.0],
        yaw=0.0,
        fuser_version=FUSER_VERSION,
    )


def _fused_lidar_only(
    scene_id: str,
    timestamp_us: int,
    det_3d: Detection3DRecord,
    projected_bbox: tuple[float, float, float, float] | None,
) -> FusedObjectRecord:
    return FusedObjectRecord(
        fusion_id=str(uuid.uuid4()),
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        fusion_type="lidar_only",
        frame_id_2d="",
        frame_id_3d=det_3d.frame_id,
        detection_id_2d="",
        detection_id_3d=det_3d.detection_id,
        class_id=det_3d.class_id,
        class_name=det_3d.class_name,
        score=det_3d.score,
        bbox_xyxy=list(projected_bbox) if projected_bbox is not None else [0.0, 0.0, 0.0, 0.0],
        center_xyz=det_3d.center_xyz,
        dimensions_whl=det_3d.dimensions_whl,
        yaw=det_3d.yaw,
        fuser_version=FUSER_VERSION,
    )


def _fuse_one_camera_frame(
    camera_frame: FrameRecord,
    detections_2d_by_frame: dict[str, list[Detection2DRecord]],
    detections_3d_by_frame: dict[str, list[Detection3DRecord]],
    calibration_by_sensor: dict[str, CalibrationRecord],
    lidar_frames_by_key: dict[tuple[str, int], list[FrameRecord]],
    iou_threshold: float,
) -> list[FusedObjectRecord]:
    """Fuse one camera frame against its synchronized lidar sweep. Safe to run in a Ray worker."""
    output: list[FusedObjectRecord] = []

    camera_detections = detections_2d_by_frame.get(camera_frame.frame_id, [])
    calibration_record = calibration_by_sensor.get(camera_frame.sensor_id)
    synchronized_lidar_frames = lidar_frames_by_key.get(
        (camera_frame.scene_id, camera_frame.timestamp_us), []
    )

    if calibration_record is None or not synchronized_lidar_frames:
        # No calibration or no synchronized lidar sweep -- nothing to fuse against.
        for det_2d in camera_detections:
            output.append(
                _fused_camera_only(camera_frame.scene_id, camera_frame.timestamp_us, det_2d)
            )
        return output

    lidar_frame = synchronized_lidar_frames[
        0
    ]  # documented simplification, see run_fusion's docstring
    lidar_detections = detections_3d_by_frame.get(lidar_frame.frame_id, [])

    projected_boxes: list[tuple[float, float, float, float]] = []
    projectable_lidar_detections: list[Detection3DRecord] = []
    for det_3d in lidar_detections:
        projected = project_box_to_bbox(
            det_3d.center_xyz, det_3d.dimensions_whl, det_3d.yaw, calibration_record
        )
        if projected is None:
            output.append(
                _fused_lidar_only(camera_frame.scene_id, camera_frame.timestamp_us, det_3d, None)
            )
            continue
        projected_boxes.append(projected)
        projectable_lidar_detections.append(det_3d)

    camera_boxes = [
        (d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]) for d in camera_detections
    ]
    matches, unmatched_lidar, unmatched_camera = associate(
        projected_boxes, camera_boxes, iou_threshold
    )

    for lidar_idx, camera_idx in matches:
        output.append(
            _fused_from_match(
                camera_frame.scene_id,
                camera_frame.timestamp_us,
                camera_detections[camera_idx],
                projectable_lidar_detections[lidar_idx],
            )
        )
    for lidar_idx in unmatched_lidar:
        output.append(
            _fused_lidar_only(
                camera_frame.scene_id,
                camera_frame.timestamp_us,
                projectable_lidar_detections[lidar_idx],
                projected_boxes[lidar_idx],
            )
        )
    for camera_idx in unmatched_camera:
        output.append(
            _fused_camera_only(
                camera_frame.scene_id, camera_frame.timestamp_us, camera_detections[camera_idx]
            )
        )

    return output


def run_fusion(
    detections_2d: list[Detection2DRecord],
    detections_3d: list[Detection3DRecord],
    frames: list[FrameRecord],
    calibration: list[CalibrationRecord],
    iou_threshold: float = 0.1,
    distributed: bool = False,
) -> list[FusedObjectRecord]:
    """Fuse camera and lidar detections, one synchronized (camera, lidar) frame pair at a time.

    Camera and lidar frames are considered synchronized when they share the
    same ``(scene_id, timestamp_us)`` — the simplifying assumption that a
    "sample" is exactly one camera capture + one lidar sweep at the same
    instant (true for how this repo's ingest fixture is built; real
    nuScenes samples are *nominally* synchronized but not bit-exact — see
    DECISIONS.md). Calibration is looked up by sensor_id, taking the first
    match if there happen to be duplicates for one channel.

    Args:
        distributed: If True, runs each camera frame's fusion via local Ray
            (see ``forge.distributed.run_distributed_map``) instead of a
            plain sequential loop — every camera frame is independent, so
            this is genuinely parallel. The shared lookup dicts are passed
            via ``shared_args`` so Ray ``ray.put()``s them once rather than
            re-serializing them into the remote function definition for
            every frame.
    """
    detections_2d_by_frame: dict[str, list[Detection2DRecord]] = defaultdict(list)
    for detection_2d in detections_2d:
        detections_2d_by_frame[detection_2d.frame_id].append(detection_2d)

    detections_3d_by_frame: dict[str, list[Detection3DRecord]] = defaultdict(list)
    for detection_3d in detections_3d:
        detections_3d_by_frame[detection_3d.frame_id].append(detection_3d)

    calibration_by_sensor: dict[str, CalibrationRecord] = {}
    for record in calibration:
        calibration_by_sensor.setdefault(record.sensor_id, record)

    camera_frames = [f for f in frames if f.sensor_id.startswith("CAM")]
    lidar_frames_by_key: dict[tuple[str, int], list[FrameRecord]] = defaultdict(list)
    for frame in frames:
        if frame.sensor_id.startswith("LIDAR"):
            lidar_frames_by_key[(frame.scene_id, frame.timestamp_us)].append(frame)

    per_frame_results = run_distributed_map(
        lambda camera_frame, det2d, det3d, calib, lidar_frames: _fuse_one_camera_frame(
            camera_frame, det2d, det3d, calib, lidar_frames, iou_threshold
        ),
        camera_frames,
        distributed=distributed,
        shared_args=(
            detections_2d_by_frame,
            detections_3d_by_frame,
            calibration_by_sensor,
            lidar_frames_by_key,
        ),
    )

    output: list[FusedObjectRecord] = []
    for frame_results in per_frame_results:
        output.extend(frame_results)
    return output
