"""Inference: load a checkpoint, run it over camera frames in the lake."""

from __future__ import annotations

import uuid
from pathlib import Path

import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN

from forge.detect2d.dataset import load_image_tensor
from forge.detect2d.model import CLASS_NAMES, build_model
from forge.logging import configure_logging, get_logger
from forge.schemas import Detection2DRecord, FrameRecord

_LOGGER_NAME = "forge.detect2d.infer"


def load_detector(
    checkpoint: Path | None, num_classes: int = len(CLASS_NAMES)
) -> tuple[FasterRCNN, str]:
    """Load a trained checkpoint, or fall back to a fresh (untrained) model.

    Returns:
        The model in eval mode, and a version string identifying its origin.
    """
    model = build_model(num_classes)
    if checkpoint is None:
        configure_logging()
        get_logger(_LOGGER_NAME).warning(
            "no_checkpoint_provided",
            note="Running with randomly initialized weights — structural smoke test only.",
        )
        model.eval()
        return model, "untrained-random-init"

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, checkpoint.stem


def run_inference(
    frames: list[FrameRecord],
    images_root: Path,
    model: FasterRCNN,
    model_version: str,
    score_threshold: float = 0.3,
    image_size: int = 320,
) -> list[Detection2DRecord]:
    """Run the detector over every camera frame and return scored detections.

    Frames whose ``sensor_id`` doesn't start with ``CAM`` are skipped (this
    detector only handles camera imagery — lidar/radar are Phase 3/5).
    Frames whose image file can't be found are skipped with a warning rather
    than aborting the whole run.
    """
    detections: list[Detection2DRecord] = []
    camera_frames = [f for f in frames if f.sensor_id.startswith("CAM")]
    configure_logging()
    logger = get_logger(_LOGGER_NAME)

    for frame in camera_frames:
        image_path = images_root / frame.data_path
        if not image_path.exists():
            logger.warning("image_not_found", frame_id=frame.frame_id, path=str(image_path))
            continue

        image = load_image_tensor(str(image_path), image_size=image_size)
        with torch.no_grad():
            prediction = model([image])[0]

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue
            class_id = int(labels[i])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
            detections.append(
                Detection2DRecord(
                    detection_id=str(uuid.uuid4()),
                    frame_id=frame.frame_id,
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    bbox_xyxy=[float(v) for v in boxes[i].tolist()],
                    model_version=model_version,
                )
            )

    return detections
