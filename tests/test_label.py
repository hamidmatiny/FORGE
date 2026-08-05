"""Tests for forge.label: trust scoring, entropy-based priority, and run_labeling."""

from __future__ import annotations

import math

import pytest

from forge.label.run import run_labeling
from forge.label.scoring import binary_entropy, compute_trust_score
from forge.schemas import Detection2DRecord, Detection3DRecord, FusedObjectRecord


def _det_2d(detection_id: str, score: float, frame_id: str = "cam0") -> Detection2DRecord:
    return Detection2DRecord(
        detection_id=detection_id,
        frame_id=frame_id,
        class_id=1,
        class_name="vehicle",
        score=score,
        bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        model_version="t",
    )


def _det_3d(detection_id: str, score: float, frame_id: str = "lidar0") -> Detection3DRecord:
    return Detection3DRecord(
        detection_id=detection_id,
        frame_id=frame_id,
        class_id=1,
        class_name="vehicle",
        score=score,
        center_xyz=[0.0, 0.0, 0.0],
        dimensions_whl=[1.0, 1.0, 1.0],
        yaw=0.0,
        model_version="t",
    )


def _fused(
    fusion_id: str,
    fusion_type: str,
    detection_id_2d: str = "",
    detection_id_3d: str = "",
) -> FusedObjectRecord:
    return FusedObjectRecord(
        fusion_id=fusion_id,
        scene_id="scene-a",
        timestamp_us=0,
        fusion_type=fusion_type,
        frame_id_2d="cam0" if detection_id_2d else "",
        frame_id_3d="lidar0" if detection_id_3d else "",
        detection_id_2d=detection_id_2d,
        detection_id_3d=detection_id_3d,
        class_id=1,
        class_name="vehicle",
        score=0.5,
        bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        center_xyz=[0.0, 0.0, 0.0],
        dimensions_whl=[1.0, 1.0, 1.0],
        yaw=0.0,
        fuser_version="t",
    )


# --- Trust scoring -----------------------------------------------------


def test_matched_trust_is_average_of_both_scores() -> None:
    trust = compute_trust_score("matched", score_2d=0.8, score_3d=0.6, single_modality_discount=0.7)
    assert trust == pytest.approx(0.7)


def test_camera_only_trust_is_discounted() -> None:
    trust = compute_trust_score(
        "camera_only", score_2d=1.0, score_3d=None, single_modality_discount=0.7
    )
    assert trust == pytest.approx(0.7)


def test_lidar_only_trust_is_discounted() -> None:
    trust = compute_trust_score(
        "lidar_only", score_2d=None, score_3d=1.0, single_modality_discount=0.5
    )
    assert trust == pytest.approx(0.5)


def test_matched_beats_single_modality_at_equal_raw_confidence() -> None:
    """The whole point of the cross-modal bonus: agreement should be worth more than discount."""
    matched = compute_trust_score(
        "matched", score_2d=0.8, score_3d=0.8, single_modality_discount=0.7
    )
    single = compute_trust_score(
        "camera_only", score_2d=0.8, score_3d=None, single_modality_discount=0.7
    )
    assert matched > single


def test_trust_score_is_clamped_to_unit_interval() -> None:
    trust = compute_trust_score("matched", score_2d=1.0, score_3d=1.0, single_modality_discount=1.0)
    assert 0.0 <= trust <= 1.0


def test_matched_missing_score_raises() -> None:
    with pytest.raises(ValueError, match="requires both"):
        compute_trust_score("matched", score_2d=0.5, score_3d=None)


def test_camera_only_missing_score_raises() -> None:
    with pytest.raises(ValueError, match="requires score_2d"):
        compute_trust_score("camera_only", score_2d=None, score_3d=0.5)


def test_unknown_fusion_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown fusion_type"):
        compute_trust_score("bogus", score_2d=0.5, score_3d=0.5)


# --- Entropy / review priority ----------------------------------------


def test_entropy_is_maximized_at_half() -> None:
    assert binary_entropy(0.5) == pytest.approx(1.0, abs=1e-6)


def test_entropy_is_zero_at_extremes() -> None:
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-3)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-3)


def test_entropy_symmetric_around_half() -> None:
    assert binary_entropy(0.3) == pytest.approx(binary_entropy(0.7), abs=1e-9)


def test_entropy_matches_hand_computed_value() -> None:
    p = 0.25
    expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert binary_entropy(p) == pytest.approx(expected)


# --- run_labeling orchestration -----------------------------------------


def test_run_labeling_routes_to_all_three_buckets() -> None:
    fused = [
        _fused("f-high", "matched", "2d-a", "3d-a"),
        _fused("f-mid", "camera_only", "2d-b"),
        _fused("f-low", "lidar_only", detection_id_3d="3d-c"),
    ]
    detections_2d = [_det_2d("2d-a", 0.9), _det_2d("2d-b", 0.55)]
    detections_3d = [_det_3d("3d-a", 0.9), _det_3d("3d-c", 0.2)]

    labels = run_labeling(fused, detections_2d, detections_3d)
    by_id = {label.fusion_id: label for label in labels}

    assert by_id["f-high"].decision == "auto_accept"
    assert by_id["f-mid"].decision == "needs_review"
    assert by_id["f-low"].decision == "rejected"


def test_run_labeling_preserves_geometry_fields() -> None:
    fused = [_fused("f-1", "matched", "2d-a", "3d-a")]
    labels = run_labeling(fused, [_det_2d("2d-a", 0.9)], [_det_3d("3d-a", 0.9)])
    assert labels[0].bbox_xyxy == fused[0].bbox_xyxy
    assert labels[0].center_xyz == fused[0].center_xyz
    assert labels[0].dimensions_whl == fused[0].dimensions_whl
    assert labels[0].class_name == fused[0].class_name


def test_run_labeling_priority_highest_for_borderline_case() -> None:
    fused = [
        _fused("f-confident", "matched", "2d-a", "3d-a"),  # trust 0.9, far from boundary
        _fused("f-borderline", "camera_only", "2d-b"),  # trust ~0.385, near the middle
    ]
    detections_2d = [_det_2d("2d-a", 0.9), _det_2d("2d-b", 0.55)]
    detections_3d = [_det_3d("3d-a", 0.9)]

    labels = run_labeling(fused, detections_2d, detections_3d)
    by_id = {label.fusion_id: label for label in labels}
    assert by_id["f-borderline"].review_priority > by_id["f-confident"].review_priority


def test_run_labeling_invalid_threshold_order_raises() -> None:
    with pytest.raises(ValueError, match="must be greater than"):
        run_labeling([], [], [], auto_accept_threshold=0.3, reject_threshold=0.7)


def test_run_labeling_empty_inputs_returns_empty() -> None:
    assert run_labeling([], [], []) == []
