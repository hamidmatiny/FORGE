"""Inference: load a checkpoint, run it over LIDAR frames in the lake."""

from __future__ import annotations

import uuid
from pathlib import Path

import torch

from forge.detect3d.model import CLASS_NAMES, NUM_QUERIES, Detector3DModule
from forge.detect3d.pointcloud import load_point_cloud
from forge.logging import configure_logging, get_logger
from forge.schemas import Detection3DRecord, FrameRecord

_LOGGER_NAME = "forge.detect3d.infer"


def load_detector(
    checkpoint: Path | None,
    num_classes: int = len(CLASS_NAMES),
    num_queries: int = NUM_QUERIES,
) -> tuple[Detector3DModule, str]:
    """Load a trained checkpoint, or fall back to a fresh (untrained) model.

    Returns:
        The model in eval mode, and a version string identifying its origin.
    """
    if checkpoint is None:
        configure_logging()
        get_logger(_LOGGER_NAME).warning(
            "no_checkpoint_provided",
            note="Running with randomly initialized weights — structural smoke test only.",
        )
        model = Detector3DModule(num_classes=num_classes, num_queries=num_queries)
        model.eval()
        return model, "untrained-random-init"

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Detector3DModule(
        num_classes=int(state["num_classes"]), num_queries=int(state["num_queries"])
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, checkpoint.stem


def run_inference(
    frames: list[FrameRecord],
    pointcloud_root: Path,
    model: Detector3DModule,
    model_version: str,
    score_threshold: float = 0.3,
) -> list[Detection3DRecord]:
    """Run the detector over every LIDAR frame and return scored 3D detections.

    Frames whose ``sensor_id`` doesn't start with ``LIDAR`` are skipped
    (camera imagery is Phase 2). Frames whose point-cloud file can't be
    found are skipped with a warning rather than aborting the whole run.
    """
    detections: list[Detection3DRecord] = []
    lidar_frames = [f for f in frames if f.sensor_id.startswith("LIDAR")]
    configure_logging()
    logger = get_logger(_LOGGER_NAME)

    for frame in lidar_frames:
        cloud_path = pointcloud_root / frame.data_path
        if not cloud_path.exists():
            logger.warning("pointcloud_not_found", frame_id=frame.frame_id, path=str(cloud_path))
            continue

        points = load_point_cloud(str(cloud_path))[:, :4]  # drop ring index
        with torch.no_grad():
            prediction = model([points])[0]  # (num_queries, 1+num_classes+7)

        num_classes = model.num_classes
        objectness = torch.sigmoid(prediction[:, 0])
        class_probs = torch.softmax(prediction[:, 1 : 1 + num_classes], dim=-1)
        box_params = prediction[:, 1 + num_classes :]

        for query_idx in range(prediction.shape[0]):
            score = float(objectness[query_idx])
            if score < score_threshold:
                continue
            class_id = int(torch.argmax(class_probs[query_idx]))
            box = box_params[query_idx]
            detections.append(
                Detection3DRecord(
                    detection_id=str(uuid.uuid4()),
                    frame_id=frame.frame_id,
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    score=score,
                    center_xyz=[float(v) for v in box[0:3].tolist()],
                    dimensions_whl=[float(v) for v in box[3:6].tolist()],
                    yaw=float(box[6]),
                    model_version=model_version,
                )
            )

    return detections
