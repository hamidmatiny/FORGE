"""Tests for forge.track: Kalman filter, association, tracker lifecycle, orchestration.

Skipped entirely when the [track] extra (numpy/scipy) isn't installed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from forge.schemas import Detection2DRecord, FrameRecord  # noqa: E402
from forge.track.association import associate, iou_batch  # noqa: E402
from forge.track.kalman import KalmanBoxTracker, bbox_to_z, state_to_bbox  # noqa: E402
from forge.track.run import run_tracking  # noqa: E402
from forge.track.tracker import SortTracker  # noqa: E402


def _det(detection_id: str, frame_id: str, x1: float, class_id: int = 1) -> Detection2DRecord:
    return Detection2DRecord(
        detection_id=detection_id,
        frame_id=frame_id,
        class_id=class_id,
        class_name="vehicle" if class_id == 1 else "pedestrian",
        score=0.9,
        bbox_xyxy=[x1, 10.0, x1 + 50.0, 60.0],
        model_version="test",
    )


def _frame(
    frame_id: str, scene_id: str, timestamp_us: int, sensor_id: str = "CAM_FRONT"
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        sensor_id=sensor_id,
        dataset_split="train",
        data_path="x",
        ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


# --- Kalman filter -----------------------------------------------------


def test_bbox_to_z_and_back_is_consistent() -> None:
    bbox = (10.0, 20.0, 60.0, 80.0)
    z = bbox_to_z(bbox)
    recovered = state_to_bbox(np.array([z[0], z[1], z[2], z[3], 0, 0, 0]))
    assert recovered == pytest.approx(bbox, abs=1e-6)


def test_kalman_predict_moves_toward_velocity() -> None:
    tracker = KalmanBoxTracker((0.0, 0.0, 50.0, 50.0))
    tracker.update((5.0, 0.0, 55.0, 50.0))  # observed a rightward shift
    tracker.x[4] = 5.0  # force a known x-velocity for a deterministic prediction
    predicted = tracker.predict()
    # center_x should have advanced further right due to velocity
    assert predicted[0] > 5.0


def test_kalman_update_pulls_state_toward_observation() -> None:
    tracker = KalmanBoxTracker((0.0, 0.0, 50.0, 50.0))
    tracker.predict()
    tracker.update((100.0, 100.0, 150.0, 150.0))
    bbox = tracker.current_bbox()
    assert bbox[0] > 10.0  # moved substantially toward the new observation


# --- Association ---------------------------------------------------------


def test_iou_batch_identical_boxes_is_one() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0)]
    iou = iou_batch(boxes, boxes)
    assert iou[0, 0] == pytest.approx(1.0)


def test_iou_batch_disjoint_boxes_is_zero() -> None:
    a = [(0.0, 0.0, 10.0, 10.0)]
    b = [(100.0, 100.0, 110.0, 110.0)]
    iou = iou_batch(a, b)
    assert iou[0, 0] == pytest.approx(0.0)


def test_associate_matches_overlapping_boxes() -> None:
    predicted = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    detections = [(1.0, 1.0, 11.0, 11.0), (101.0, 101.0, 111.0, 111.0)]
    matches, unmatched_tracks, unmatched_dets = associate(predicted, detections, iou_threshold=0.3)
    assert len(matches) == 2
    assert unmatched_tracks == []
    assert unmatched_dets == []


def test_associate_below_threshold_stays_unmatched() -> None:
    predicted = [(0.0, 0.0, 10.0, 10.0)]
    detections = [(50.0, 50.0, 60.0, 60.0)]  # no overlap at all
    matches, unmatched_tracks, unmatched_dets = associate(predicted, detections, iou_threshold=0.3)
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_associate_empty_inputs() -> None:
    matches, unmatched_tracks, unmatched_dets = associate([], [(0.0, 0.0, 1.0, 1.0)], 0.3)
    assert matches == []
    assert unmatched_tracks == []
    assert unmatched_dets == [0]


# --- SortTracker lifecycle -------------------------------------------------


def test_tracker_assigns_same_id_to_drifting_object() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=3)
    track_ids = []
    for i in range(4):
        results = tracker.step([_det(f"d{i}", f"f{i}", x1=10.0 + i * 5)])
        track_ids.append(results[0][1])
    assert len(set(track_ids)) == 1  # same track the whole time


def test_tracker_retires_after_max_age_missed_frames() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=2)
    tracker.step([_det("d0", "f0", x1=10.0)])
    assert len(tracker.tracks) == 1
    tracker.step([])  # miss 1
    assert len(tracker.tracks) == 1
    tracker.step([])  # miss 2 (== max_age, still alive)
    assert len(tracker.tracks) == 1
    tracker.step([])  # miss 3 (> max_age, retired)
    assert len(tracker.tracks) == 0


def test_tracker_gives_new_id_after_retirement() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=1)
    r0 = tracker.step([_det("d0", "f0", x1=10.0)])
    first_id = r0[0][1]
    tracker.step([])
    tracker.step([])  # exceeds max_age=1, retires
    r3 = tracker.step([_det("d3", "f3", x1=200.0)])
    assert r3[0][1] != first_id


def test_tracker_hit_streak_increments_on_matches() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=3)
    ages = []
    for i in range(3):
        results = tracker.step([_det(f"d{i}", f"f{i}", x1=10.0 + i * 2)])
        ages.append(results[0][2])
    assert ages == [1, 2, 3]


def test_tracker_handles_two_simultaneous_objects() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=3)
    results = tracker.step([_det("d0", "f0", x1=10.0), _det("d1", "f0", x1=500.0, class_id=2)])
    assert len({r[1] for r in results}) == 2  # two distinct new tracks


# --- run_tracking orchestration -------------------------------------------


def test_run_tracking_single_scene_single_object() -> None:
    frames = [_frame(f"f{i}", "scene-a", i * 100_000) for i in range(3)]
    detections = [_det(f"d{i}", f"f{i}", x1=10.0 + i * 3) for i in range(3)]
    tracks = run_tracking(detections, frames, iou_threshold=0.3, max_age=3)
    assert len(tracks) == 3
    assert len({t.track_id for t in tracks}) == 1


def test_run_tracking_separates_different_scenes() -> None:
    frames = [_frame("f0", "scene-a", 0), _frame("f1", "scene-b", 0)]
    detections = [_det("d0", "f0", x1=10.0), _det("d1", "f1", x1=10.0)]
    tracks = run_tracking(detections, frames, iou_threshold=0.3, max_age=3)
    # Same bbox, but different scenes -> must be different tracks, never merged.
    assert len({t.track_id for t in tracks}) == 2


def test_run_tracking_skips_non_camera_frames() -> None:
    frames = [_frame("f0", "scene-a", 0, sensor_id="LIDAR_TOP")]
    detections: list[Detection2DRecord] = []  # detect2d never produces lidar detections
    tracks = run_tracking(detections, frames)
    assert tracks == []


def test_run_tracking_empty_frame_ages_out_track() -> None:
    frames = [_frame(f"f{i}", "scene-a", i * 100_000) for i in range(5)]
    detections = [_det("d0", "f0", x1=10.0)]  # only the first frame has a detection
    tracks = run_tracking(detections, frames, iou_threshold=0.3, max_age=1)
    assert len(tracks) == 1
    assert tracks[0].track_age == 1
