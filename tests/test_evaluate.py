"""Tests for forge.evaluate: GT ingestion, detection metrics, and evaluation orchestration.

GT ingestion and orchestration tests need no extras (pure schemas + stdlib
math). The MLflow/W&B logging tests are skipped if the [evaluate] extra
isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.evaluate.ingest_gt import ingest_ground_truth, quaternion_to_yaw
from forge.evaluate.metrics import evaluate_class
from forge.evaluate.run import run_evaluation
from forge.schemas import PseudoLabelRecord

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "nuscenes_mini_synthetic"


def _label(
    pseudo_label_id: str,
    fusion_type: str,
    class_name: str,
    center_xyz: list[float],
    decision: str = "auto_accept",
    trust_score: float = 0.9,
) -> PseudoLabelRecord:
    return PseudoLabelRecord(
        pseudo_label_id=pseudo_label_id,
        fusion_id=f"f-{pseudo_label_id}",
        scene_id="scene-a",
        timestamp_us=0,
        fusion_type=fusion_type,
        class_id=1,
        class_name=class_name,
        bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        center_xyz=center_xyz,
        dimensions_whl=[1.0, 1.0, 1.0],
        yaw=0.0,
        trust_score=trust_score,
        decision=decision,
        review_priority=0.1,
        labeler_version="t",
    )


# --- GT ingestion -----------------------------------------------------


def test_quaternion_to_yaw_identity_is_zero() -> None:
    assert quaternion_to_yaw([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_quaternion_to_yaw_ninety_degrees() -> None:
    import math

    # 90-degree rotation about z: w=cos(45deg), z=sin(45deg).
    half = math.pi / 4
    q = [math.cos(half), 0.0, 0.0, math.sin(half)]
    assert quaternion_to_yaw(q) == pytest.approx(math.pi / 2, abs=1e-6)


def test_ingest_ground_truth_from_fixture() -> None:
    records = ingest_ground_truth(FIXTURE_ROOT)
    assert len(records) == 4
    scene_names = {r.scene_id for r in records}
    assert scene_names == {"scene-0001", "scene-0002"}
    assert all(len(r.center_xyz) == 3 for r in records)
    assert all(len(r.dimensions_whl) == 3 for r in records)


def test_ingest_ground_truth_missing_table_raises(tmp_path: Path) -> None:
    version_dir = tmp_path / "v1.0-mini"
    version_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="scene.json"):
        ingest_ground_truth(tmp_path)


# --- Detection metrics --------------------------------------------------


def test_evaluate_class_perfect_match() -> None:
    gt = [(0.0, 0.0), (10.0, 10.0)]
    predictions = [((0.1, 0.1), 0.9), ((10.1, 10.1), 0.8)]
    result = evaluate_class(predictions, gt, distance_threshold_m=1.0)
    assert result.num_matched == 2
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)


def test_evaluate_class_no_predictions() -> None:
    result = evaluate_class([], [(0.0, 0.0)], distance_threshold_m=1.0)
    assert result.num_matched == 0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_evaluate_class_no_ground_truth() -> None:
    result = evaluate_class([((0.0, 0.0), 0.9)], [], distance_threshold_m=1.0)
    assert result.num_matched == 0
    assert result.recall == 0.0
    assert result.average_precision == 0.0


def test_evaluate_class_textbook_ap() -> None:
    """Known textbook AP for a TP,FP,TP,FP score-ordered pattern with 2 GT is 0.833."""
    gt = [(0.0, 0.0), (10.0, 10.0)]
    predictions = [
        ((0.1, 0.1), 0.9),
        ((50.0, 50.0), 0.8),
        ((10.1, 10.1), 0.7),
        ((60.0, 60.0), 0.6),
    ]
    result = evaluate_class(predictions, gt, distance_threshold_m=1.0)
    assert result.average_precision == pytest.approx(0.8333, abs=0.001)


def test_evaluate_class_two_predictions_cannot_double_claim_one_gt() -> None:
    gt = [(0.0, 0.0)]
    predictions = [((0.05, 0.05), 0.9), ((0.06, 0.06), 0.8)]  # both near the one GT
    result = evaluate_class(predictions, gt, distance_threshold_m=1.0)
    assert result.num_matched == 1  # only one can claim it, higher-score wins


# --- run_evaluation orchestration -----------------------------------------


def test_run_evaluation_filters_by_decision() -> None:
    labels = [
        _label("1", "matched", "vehicle.car", [0.0, 0.0, 0.0], decision="auto_accept"),
        _label("2", "matched", "vehicle.car", [50.0, 50.0, 0.0], decision="needs_review"),
    ]
    gt = ingest_ground_truth(FIXTURE_ROOT)
    metrics_auto_only = run_evaluation(labels, gt, decision_filter="auto_accept")
    overall = next(m for m in metrics_auto_only if m.class_name == "overall")
    assert overall.num_predictions == 1  # only the auto_accept one counted

    metrics_all = run_evaluation(labels, gt, decision_filter="all")
    overall_all = next(m for m in metrics_all if m.class_name == "overall")
    assert overall_all.num_predictions == 2


def test_run_evaluation_excludes_camera_only() -> None:
    labels = [
        _label("1", "camera_only", "vehicle.car", [0.0, 0.0, 0.0]),  # no real 3D center
        _label("2", "lidar_only", "vehicle.car", [10.0, 0.0, 0.0]),
    ]
    gt = ingest_ground_truth(FIXTURE_ROOT)
    metrics = run_evaluation(labels, gt, decision_filter="all")
    overall = next(m for m in metrics if m.class_name == "overall")
    assert overall.num_predictions == 1  # camera_only excluded


def test_run_evaluation_produces_one_row_per_class_plus_overall() -> None:
    labels = [_label("1", "matched", "vehicle.car", [10.5, 0.2, 0.0])]
    gt = ingest_ground_truth(FIXTURE_ROOT)
    metrics = run_evaluation(labels, gt, decision_filter="all")
    class_names = {m.class_name for m in metrics}
    assert "overall" in class_names
    assert "vehicle.car" in class_names
    assert "human.pedestrian.adult" in class_names


def test_run_evaluation_same_run_id_across_all_rows() -> None:
    labels = [_label("1", "matched", "vehicle.car", [10.5, 0.2, 0.0])]
    gt = ingest_ground_truth(FIXTURE_ROOT)
    metrics = run_evaluation(labels, gt, decision_filter="all")
    run_ids = {m.eval_run_id for m in metrics}
    assert len(run_ids) == 1


def test_run_evaluation_empty_inputs() -> None:
    metrics = run_evaluation([], [], decision_filter="all")
    assert len(metrics) == 1
    assert metrics[0].class_name == "overall"
    assert metrics[0].num_gt == 0


# --- MLflow / W&B logging (skipped without the [evaluate] extra) -----------


def test_log_to_mlflow_creates_local_sqlite_store(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    from forge.evaluate.tracking import log_to_mlflow

    log_to_mlflow(tmp_path, "test-run", {"threshold": 2.0}, {"precision": 0.8})
    assert (tmp_path / "mlflow" / "mlflow.db").exists()


def test_log_to_wandb_creates_local_offline_dir(tmp_path: Path) -> None:
    pytest.importorskip("wandb")
    from forge.evaluate.tracking import log_to_wandb

    log_to_wandb(tmp_path, "test-run", {"threshold": 2.0}, {"precision": 0.8})
    assert (tmp_path / "wandb").exists()


def test_log_to_mlflow_missing_package_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from forge.evaluate.tracking import log_to_mlflow

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlflow":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    log_to_mlflow(tmp_path, "test-run", {}, {})  # should not raise
