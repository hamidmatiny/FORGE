"""Ground-truth ingestion: nuScenes sample_annotation -> the ground_truth table.

Eval-only, mirroring forge.ingest.nuscenes but for a different purpose:
these annotations are never a pipeline input (see the dataset notice in
README.md), only something `forge evaluate` scores auto-labels against.

Deliberately simplified from the real nuScenes schema (see DECISIONS.md):
- Real nuScenes stores category through instance.json -> category.json
  joins; this expects ``category_name`` flattened directly onto each
  annotation row instead, since GT is eval-only and the extra join adds
  complexity without changing anything the pipeline itself does.
- ``size_whl`` uses FORGE's own [width, height, length] convention
  (matching ``detections_3d``), not nuScenes' native [w, l, h] order.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from forge.schemas import GroundTruthRecord


class _RawScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    name: str


class _RawSample(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    scene_token: str
    timestamp: int


class _RawAnnotation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    token: str
    sample_token: str
    category_name: str
    translation: list[float]
    size: list[float]
    rotation: list[float]
    num_lidar_pts: int


def quaternion_to_yaw(quaternion: list[float]) -> float:
    """[w, x, y, z] -> heading angle (radians) about the vertical (z) axis.

    Standard closed-form yaw extraction, assuming the object's roll/pitch
    are negligible (the normal assumption for ground-vehicle traffic
    participants in a BEV annotation).
    """
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _load_table(root: Path, name: str) -> list[dict[str, object]]:
    path = root / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected ground-truth table '{name}.json' under {root}. Confirm "
            "--gt-input-dir points at a directory containing '<version>/*.json'."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array.")
    return data


def ingest_ground_truth(input_dir: Path, version: str = "v1.0-mini") -> list[GroundTruthRecord]:
    """Parse nuScenes-format sample_annotation.json into GroundTruthRecords.

    Args:
        input_dir: Dataset root; must contain ``<version>/*.json``.
        version: nuScenes version directory name.

    Raises:
        FileNotFoundError: If a required JSON table is missing.
    """
    table_root = input_dir / version
    scenes = {s["token"]: _RawScene.model_validate(s) for s in _load_table(table_root, "scene")}
    samples = {s["token"]: _RawSample.model_validate(s) for s in _load_table(table_root, "sample")}
    annotations = [
        _RawAnnotation.model_validate(a) for a in _load_table(table_root, "sample_annotation")
    ]

    records: list[GroundTruthRecord] = []
    for annotation in annotations:
        sample = samples[annotation.sample_token]
        scene_name = scenes[sample.scene_token].name
        records.append(
            GroundTruthRecord(
                annotation_id=annotation.token,
                scene_id=scene_name,
                timestamp_us=sample.timestamp,
                category_name=annotation.category_name,
                center_xyz=annotation.translation,
                dimensions_whl=annotation.size,
                yaw=quaternion_to_yaw(annotation.rotation),
                num_lidar_pts=annotation.num_lidar_pts,
            )
        )

    return records
