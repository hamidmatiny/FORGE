"""Tests for detect2d: model, training loop, and inference.

Skipped entirely when the [detect2d] extra (torch/torchvision/lightning)
isn't installed, matching how the CLI itself degrades gracefully.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("lightning")

from forge.detect2d.dataset import (  # noqa: E402
    SyntheticDetectionDataset,
    detection_collate,
    load_image_tensor,
)
from forge.detect2d.infer import load_detector, run_inference  # noqa: E402
from forge.detect2d.model import CLASS_NAMES, Detector2DModule, build_model  # noqa: E402
from forge.detect2d.train import train_detector  # noqa: E402
from forge.schemas import FrameRecord  # noqa: E402

FIXTURE_IMAGES_ROOT = Path(__file__).parent / "fixtures" / "nuscenes_mini_synthetic"


def test_synthetic_dataset_shapes() -> None:
    dataset = SyntheticDetectionDataset(num_samples=4, image_size=64, num_classes=3, seed=1)
    assert len(dataset) == 4
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target["boxes"].shape[1] == 4
    assert target["boxes"].shape[0] == target["labels"].shape[0]
    assert target["labels"].min() >= 1
    assert target["labels"].max() <= 2


def test_detection_collate_returns_lists() -> None:
    dataset = SyntheticDetectionDataset(num_samples=3, image_size=32, seed=2)
    batch = [dataset[i] for i in range(3)]
    images, targets = detection_collate(batch)
    assert len(images) == 3
    assert len(targets) == 3
    assert all(isinstance(img, torch.Tensor) for img in images)


def test_build_model_runs_train_and_eval_forward() -> None:
    model = build_model(num_classes=3)
    model.train()
    images = [torch.rand(3, 64, 64)]
    targets = [{"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([1])}]
    losses = model(images, targets)
    assert "loss_classifier" in losses
    assert all(torch.isfinite(v) for v in losses.values())

    model.eval()
    with torch.no_grad():
        preds = model([torch.rand(3, 64, 64)])
    assert {"boxes", "labels", "scores"} <= preds[0].keys()


def test_lightning_module_training_step() -> None:
    module = Detector2DModule(num_classes=3, lr=1e-4)
    dataset = SyntheticDetectionDataset(num_samples=2, image_size=64, num_classes=3, seed=3)
    batch = detection_collate([dataset[0], dataset[1]])
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_train_detector_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    final_loss = train_detector(
        output_checkpoint=checkpoint,
        max_steps=2,
        num_classes=3,
        num_samples=4,
        batch_size=2,
    )
    assert checkpoint.exists()
    assert final_loss == final_loss  # not NaN

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "model_state_dict" in state
    assert state["num_classes"] == 3


def test_load_detector_without_checkpoint_returns_untrained_model() -> None:
    model, version = load_detector(checkpoint=None, num_classes=3)
    assert version == "untrained-random-init"
    model.eval()
    with torch.no_grad():
        preds = model([torch.rand(3, 64, 64)])
    assert "boxes" in preds[0]


def test_load_detector_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    train_detector(output_checkpoint=checkpoint, max_steps=1, num_classes=3, num_samples=2)
    model, version = load_detector(checkpoint, num_classes=3)
    assert version == "ckpt"
    model.eval()


def test_load_image_tensor_from_fixture() -> None:
    image_path = FIXTURE_IMAGES_ROOT / "samples" / "CAM_FRONT" / "scene0001_00.jpg"
    tensor = load_image_tensor(str(image_path), image_size=64)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32


def test_run_inference_over_fixture_frames() -> None:
    frames = [
        FrameRecord(
            frame_id="sd-1-cam",
            scene_id="scene-0001",
            timestamp_us=1,
            sensor_id="CAM_FRONT",
            dataset_split="mini_train",
            data_path="samples/CAM_FRONT/scene0001_00.jpg",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        FrameRecord(
            frame_id="sd-1-lidar",
            scene_id="scene-0001",
            timestamp_us=1,
            sensor_id="LIDAR_TOP",
            dataset_split="mini_train",
            data_path="samples/LIDAR_TOP/scene0001_00.pcd.bin",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    model, version = load_detector(checkpoint=None, num_classes=len(CLASS_NAMES))
    detections = run_inference(
        frames, FIXTURE_IMAGES_ROOT, model, version, score_threshold=0.0, image_size=64
    )
    # Only the CAM_FRONT frame should be processed (lidar skipped, and it has
    # no real file on disk anyway); every kept detection references it.
    assert len(detections) > 0
    assert all(d.frame_id == "sd-1-cam" for d in detections)
    assert all(0.0 <= d.score <= 1.0 for d in detections)
    assert all(len(d.bbox_xyxy) == 4 for d in detections)


def test_run_inference_skips_missing_image_file() -> None:
    frames = [
        FrameRecord(
            frame_id="missing-frame",
            scene_id="scene-x",
            timestamp_us=1,
            sensor_id="CAM_FRONT",
            dataset_split="mini_train",
            data_path="samples/CAM_FRONT/does_not_exist.jpg",
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    model, version = load_detector(checkpoint=None, num_classes=3)
    detections = run_inference(frames, FIXTURE_IMAGES_ROOT, model, version)
    assert detections == []
