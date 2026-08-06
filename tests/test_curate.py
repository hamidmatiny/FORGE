"""Tests for forge.curate: geometric feature vectors and LanceDB dedup.

Skipped entirely when the [curate] extra (lancedb) isn't installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from forge.curate.features import build_feature_vector  # noqa: E402
from forge.curate.run import run_curation  # noqa: E402
from forge.schemas import PseudoLabelRecord  # noqa: E402


def _label(
    pseudo_label_id: str,
    scene_id: str,
    class_name: str,
    center_xyz: list[float],
    trust_score: float,
    dimensions_whl: list[float] | None = None,
    yaw: float = 0.0,
    decision: str = "auto_accept",
) -> PseudoLabelRecord:
    return PseudoLabelRecord(
        pseudo_label_id=pseudo_label_id,
        fusion_id=f"f-{pseudo_label_id}",
        scene_id=scene_id,
        timestamp_us=0,
        fusion_type="matched",
        class_id=1,
        class_name=class_name,
        bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        center_xyz=center_xyz,
        dimensions_whl=dimensions_whl or [1.0, 1.0, 1.0],
        yaw=yaw,
        trust_score=trust_score,
        decision=decision,
        review_priority=0.1,
        labeler_version="t",
    )


# --- Feature vectors -----------------------------------------------------


def test_build_feature_vector_shape_and_values() -> None:
    vector = build_feature_vector([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], yaw=0.0)
    assert len(vector) == 8
    assert vector[:6] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert vector[6] == pytest.approx(0.0)  # sin(0)
    assert vector[7] == pytest.approx(1.0)  # cos(0)


def test_build_feature_vector_yaw_wraparound_is_close_in_feature_space() -> None:
    """-pi and +pi are the same heading -- sin/cos encoding should keep them close."""
    v1 = build_feature_vector([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], yaw=math.pi)
    v2 = build_feature_vector([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], yaw=-math.pi)
    assert v1[6] == pytest.approx(v2[6], abs=1e-6)
    assert v1[7] == pytest.approx(v2[7], abs=1e-6)


# --- run_curation orchestration -----------------------------------------


def test_run_curation_flags_near_duplicate(tmp_path: Path) -> None:
    labels = [
        _label("high", "scene-a", "vehicle", [10.0, 0.0, 0.0], trust_score=0.9),
        _label("low-dup", "scene-a", "vehicle", [10.1, 0.05, 0.0], trust_score=0.7),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    by_id = {c.pseudo_label_id: c for c in curated}

    assert not by_id["high"].is_duplicate
    assert by_id["low-dup"].is_duplicate
    assert by_id["low-dup"].duplicate_of_id == "high"


def test_run_curation_keeps_distinct_objects_separate(tmp_path: Path) -> None:
    labels = [
        _label("car", "scene-a", "vehicle", [10.0, 0.0, 0.0], trust_score=0.9),
        _label("pedestrian", "scene-a", "pedestrian", [5.0, 3.0, 0.0], trust_score=0.8),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    assert all(not c.is_duplicate for c in curated)


def test_run_curation_same_location_different_class_not_a_duplicate(tmp_path: Path) -> None:
    """Dedup must never cross class boundaries, even at identical coordinates."""
    labels = [
        _label("car", "scene-a", "vehicle", [10.0, 0.0, 0.0], trust_score=0.9),
        _label("pedestrian-same-spot", "scene-a", "pedestrian", [10.0, 0.0, 0.0], trust_score=0.8),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    assert all(not c.is_duplicate for c in curated)


def test_run_curation_same_location_different_scene_not_a_duplicate(tmp_path: Path) -> None:
    """Dedup must never cross scene boundaries either."""
    labels = [
        _label("car-scene-a", "scene-a", "vehicle", [10.0, 0.0, 0.0], trust_score=0.9),
        _label("car-scene-b", "scene-b", "vehicle", [10.0, 0.0, 0.0], trust_score=0.8),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    assert all(not c.is_duplicate for c in curated)


def test_run_curation_far_apart_objects_not_duplicates(tmp_path: Path) -> None:
    labels = [
        _label("a", "scene-a", "vehicle", [0.0, 0.0, 0.0], trust_score=0.9),
        _label("b", "scene-a", "vehicle", [100.0, 100.0, 0.0], trust_score=0.8),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    assert all(not c.is_duplicate for c in curated)


def test_run_curation_higher_trust_always_wins_regardless_of_input_order(tmp_path: Path) -> None:
    labels = [
        _label("low", "scene-a", "vehicle", [10.0, 0.0, 0.0], trust_score=0.5),
        _label("high", "scene-a", "vehicle", [10.05, 0.0, 0.0], trust_score=0.95),
    ]
    curated = run_curation(labels, tmp_path / "lancedb", distance_threshold=1.0)
    by_id = {c.pseudo_label_id: c for c in curated}
    assert not by_id["high"].is_duplicate
    assert by_id["low"].is_duplicate
    assert by_id["low"].duplicate_of_id == "high"


def test_run_curation_filters_by_decision(tmp_path: Path) -> None:
    labels = [
        _label("accepted", "scene-a", "vehicle", [0.0, 0.0, 0.0], trust_score=0.9),
        _label(
            "review",
            "scene-a",
            "vehicle",
            [50.0, 50.0, 0.0],
            trust_score=0.5,
            decision="needs_review",
        ),
    ]
    curated_default = run_curation(labels, tmp_path / "lancedb1")
    assert len(curated_default) == 1
    assert curated_default[0].pseudo_label_id == "accepted"

    curated_all = run_curation(labels, tmp_path / "lancedb2", decision_filter="all")
    assert len(curated_all) == 2


def test_run_curation_empty_input(tmp_path: Path) -> None:
    assert run_curation([], tmp_path / "lancedb") == []


def test_run_curation_preserves_geometry_fields(tmp_path: Path) -> None:
    labels = [_label("a", "scene-a", "vehicle", [1.0, 2.0, 3.0], trust_score=0.9)]
    curated = run_curation(labels, tmp_path / "lancedb")
    assert curated[0].center_xyz == [1.0, 2.0, 3.0]
    assert curated[0].class_name == "vehicle"
