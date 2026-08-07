"""Run tracking over every (scene, sensor) sequence in the lake."""

from __future__ import annotations

from collections import defaultdict

from forge.distributed import run_distributed_map
from forge.schemas import Detection2DRecord, FrameRecord, TrackRecord
from forge.track.tracker import SortTracker

TRACKER_VERSION = "sort-v1"


def _track_one_group(
    group: tuple[tuple[str, str], list[FrameRecord]],
    detections_by_frame: dict[str, list[Detection2DRecord]],
    iou_threshold: float,
    max_age: int,
) -> list[TrackRecord]:
    """Run one SortTracker over one (scene, sensor) group. Safe to run in a Ray worker."""
    (scene_id, sensor_id), group_frames = group
    ordered_frames = sorted(group_frames, key=lambda f: f.timestamp_us)
    tracker = SortTracker(iou_threshold=iou_threshold, max_age=max_age)

    output: list[TrackRecord] = []
    for frame in ordered_frames:
        frame_detections = detections_by_frame.get(frame.frame_id, [])
        step_results = tracker.step(frame_detections)

        for detection, local_track_id, hit_streak in step_results:
            # SortTracker's counter is local to one (scene, sensor) instance, so
            # "track-000001" would otherwise collide across different groups —
            # scope it globally by prefixing with the group it came from.
            global_track_id = f"{scene_id}:{sensor_id}:{local_track_id}"
            output.append(
                TrackRecord(
                    track_id=global_track_id,
                    detection_id=detection.detection_id,
                    frame_id=detection.frame_id,
                    scene_id=scene_id,
                    sensor_id=sensor_id,
                    timestamp_us=frame.timestamp_us,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    bbox_xyxy=detection.bbox_xyxy,
                    score=detection.score,
                    track_age=hit_streak,
                    tracker_version=TRACKER_VERSION,
                )
            )
    return output


def run_tracking(
    detections: list[Detection2DRecord],
    frames: list[FrameRecord],
    iou_threshold: float = 0.3,
    max_age: int = 3,
    distributed: bool = False,
) -> list[TrackRecord]:
    """Associate 2D detections into tracks, one independent tracker per (scene, sensor).

    Tracks never span scenes or sensors — a fresh ``SortTracker`` is created
    for each ``(scene_id, sensor_id)`` group, stepped once per frame in that
    group in timestamp order (including frames with zero detections, so
    stale tracks age out correctly rather than lingering forever). Groups
    are fully independent of each other, so this is genuinely parallel —
    unlike ``curate``'s incremental LanceDB writes (see its docstring),
    nothing here depends on another group's output.

    Args:
        distributed: If True, runs each (scene, sensor) group's tracker via
            local Ray (see ``forge.distributed.run_distributed_map``)
            instead of a plain sequential loop. Same results either way.
            ``detections_by_frame`` is passed via ``shared_args`` so Ray
            ``ray.put()``s it once rather than re-serializing it into the
            remote function definition for every group.
    """
    detections_by_frame: dict[str, list[Detection2DRecord]] = defaultdict(list)
    for detection in detections:
        detections_by_frame[detection.frame_id].append(detection)

    frames_by_group: dict[tuple[str, str], list[FrameRecord]] = defaultdict(list)
    for frame in frames:
        if not frame.sensor_id.startswith("CAM"):
            continue  # detections_2d only ever comes from camera frames
        frames_by_group[(frame.scene_id, frame.sensor_id)].append(frame)

    per_group_results = run_distributed_map(
        lambda group, det_by_frame: _track_one_group(group, det_by_frame, iou_threshold, max_age),
        list(frames_by_group.items()),
        distributed=distributed,
        shared_args=(detections_by_frame,),
    )

    output: list[TrackRecord] = []
    for group_tracks in per_group_results:
        output.extend(group_tracks)
    return output
