"""Tests for detect3d: point cloud loading, model, training loop, and inference.

Skipped entirely when the [detect3d] extra (torch/lightning/numpy) isn't
installed, matching how the CLI itself degrades gracefully.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
np = pytest.importorskip("numpy")

from forge.detect3d.dataset import SyntheticPointCloudDataset, point3d_collate  # noqa: E402
from forge.detect3d.infer import load_detector, run_inference  # noqa: E402
from forge.detect3d.model import (  # noqa: E402
    BOX_DIM,
    CLASS_NAMES,
    NUM_QUERIES,
    Detector3DModule,
    PointNetEncoder,
)
from forge.detect3d.pointcloud import load_point_cloud  # noqa: E402
from forge.detect3d.train import train_detector  # noqa: E402
from forge.schemas import FrameRecord  # noqa: E402

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "nuscenes_mini_synthetic"


def test_point_cloud_loader_reads_fixture() -> None:
    path = FIXTURE_ROOT / "samples" / "LIDAR_TOP" / "scene0001_00.pcd.bin"
    points = load_point_cloud(str(path))
    assert points.shape == (150, 5)
    assert points.dtype == torch.float32


def test_point_cloud_loader_rejects_bad_size(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pcd.bin"
    np.zeros(7, dtype=np.float32).tofile(bad_path)
    with pytest.raises(ValueError, match="not a multiple"):
        load_point_cloud(str(bad_path))


def test_pointnet_encoder_permutation_invariant() -> None:
    encoder = PointNetEncoder(in_features=4, out_features=32)
    points = torch.rand(20, 4)
    feature_a = encoder(points)
    perm = torch.randperm(20)
    feature_b = encoder(points[perm])
    assert torch.allclose(feature_a, feature_b, atol=1e-5)


def test_synthetic_pointcloud_dataset_shapes() -> None:
    dataset = SyntheticPointCloudDataset(num_samples=3, num_points=50, num_queries=4, seed=1)
    assert len(dataset) == 3
    points, classes, boxes = dataset[0]
    assert points.shape == (50, 4)
    assert classes.shape == (4,)
    assert boxes.shape == (4, BOX_DIM)


def test_point3d_collate_batches_correctly() -> None:
    dataset = SyntheticPointCloudDataset(num_samples=2, num_points=30, seed=2)
    batch = [dataset[0], dataset[1]]
    point_clouds, classes, boxes = point3d_collate(batch)
    assert len(point_clouds) == 2
    assert classes.shape == (2, NUM_QUERIES)
    assert boxes.shape == (2, NUM_QUERIES, BOX_DIM)


def test_detector_forward_shape() -> None:
    module = Detector3DModule(num_classes=3, num_queries=4)
    clouds = [torch.rand(40, 4), torch.rand(60, 4)]
    output = module(clouds)
    assert output.shape == (2, 4, 1 + 3 + BOX_DIM)


def test_detector_handles_variable_point_counts() -> None:
    """Different clouds can have a different number of points in the same batch."""
    module = Detector3DModule(num_classes=3, num_queries=4)
    clouds = [torch.rand(10, 4), torch.rand(500, 4)]
    output = module(clouds)
    assert output.shape == (2, 4, 1 + 3 + BOX_DIM)
    assert torch.isfinite(output).all()


def test_training_step_produces_finite_loss() -> None:
    module = Detector3DModule(num_classes=3, num_queries=4)
    dataset = SyntheticPointCloudDataset(num_samples=2, num_points=30, seed=3)
    batch = point3d_collate([dataset[0], dataset[1]])
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_train_detector_smoke_and_loss_decreases(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    early_loss = train_detector(
        output_checkpoint=checkpoint, max_steps=2, num_samples=4, batch_size=2, seed=0
    )
    later_loss = train_detector(
        output_checkpoint=checkpoint, max_steps=40, num_samples=4, batch_size=2, lr=1e-2, seed=0
    )
    assert checkpoint.exists()
    assert early_loss == early_loss  # not NaN
    assert later_loss < early_loss  # confirms real gradient signal, not a no-op

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "model_state_dict" in state
    assert state["num_queries"] == NUM_QUERIES


def test_load_detector_without_checkpoint_returns_untrained_model() -> None:
    model, version = load_detector(checkpoint=None, num_classes=3)
    assert version == "untrained-random-init"
    with torch.no_grad():
        output = model([torch.rand(20, 4)])
    assert output.shape[1] == NUM_QUERIES


def test_load_detector_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    train_detector(output_checkpoint=checkpoint, max_steps=1, num_samples=2)
    model, version = load_detector(checkpoint)
    assert version == "ckpt"


def test_run_inference_over_fixture_frames() -> None:
    frames = [
        FrameRecord(
            frame_id="sd-1-lidar",
            scene_id="scene-0001",
            timestamp_us=1,
            sensor_id="LIDAR_TOP",
            dataset_split="mini_train",
            data_path="samples/LIDAR_TOP/scene0001_00.pcd.bin",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        FrameRecord(
            frame_id="sd-1-cam",
            scene_id="scene-0001",
            timestamp_us=1,
            sensor_id="CAM_FRONT",
            dataset_split="mini_train",
            data_path="samples/CAM_FRONT/scene0001_00.jpg",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    model, version = load_detector(checkpoint=None, num_classes=len(CLASS_NAMES))
    detections = run_inference(frames, FIXTURE_ROOT, model, version, score_threshold=0.0)
    assert len(detections) == NUM_QUERIES  # only the LIDAR frame is processed
    assert all(d.frame_id == "sd-1-lidar" for d in detections)
    assert all(len(d.center_xyz) == 3 for d in detections)
    assert all(len(d.dimensions_whl) == 3 for d in detections)


def test_run_inference_skips_missing_pointcloud_file() -> None:
    frames = [
        FrameRecord(
            frame_id="missing-frame",
            scene_id="scene-x",
            timestamp_us=1,
            sensor_id="LIDAR_TOP",
            dataset_split="mini_train",
            data_path="samples/LIDAR_TOP/does_not_exist.pcd.bin",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    model, version = load_detector(checkpoint=None, num_classes=3)
    detections = run_inference(frames, FIXTURE_ROOT, model, version)
    assert detections == []
