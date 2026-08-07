"""Write pseudo-label 3D boxes to a rerun .rrd file (headless, no live viewer)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import rerun as rr
from rerun.recording_stream import RecordingStream

from forge.schemas import PseudoLabelRecord

# camera_only pseudo-labels carry a sentinel center_xyz/dimensions_whl/yaw of
# all zeros (see fused_objects/pseudo_labels schemas) since they have no real
# 3D grounding. Logging them as Boxes3D at the origin would stack hundreds of
# meaningless boxes at [0,0,0] and mislead anyone reviewing the scene in rerun.
# MCAP export keeps them (2D bbox_xyxy is still meaningful there); rerun export
# skips them entirely. Same reasoning as forge.curate's geometric dedup exclusion
# and forge.evaluate's _3D_GROUNDED_FUSION_TYPES filter.
_CAMERA_ONLY_SENTINEL_CENTER = (0.0, 0.0, 0.0)
_CAMERA_ONLY_SENTINEL_DIMS = (0.0, 0.0, 0.0)


def _include_in_rerun(record: PseudoLabelRecord) -> bool:
    if record.fusion_type == "camera_only":
        return False
    center = tuple(record.center_xyz)
    dims = tuple(record.dimensions_whl)
    return not (center == _CAMERA_ONLY_SENTINEL_CENTER and dims == _CAMERA_ONLY_SENTINEL_DIMS)


def _filter_labels(
    labels: list[PseudoLabelRecord],
    decision_filter: str,
) -> list[PseudoLabelRecord]:
    if decision_filter == "all":
        return labels
    return [label for label in labels if label.decision == decision_filter]


def build_rerun_recording(
    labels: list[PseudoLabelRecord],
    path: Path | str,
    *,
    decision_filter: str = "auto_accept",
    application_id: str = "forge-visualize",
) -> int:
    """Log 3D boxes per timestamp to a rerun recording file.

    Returns the number of 3D boxes written (camera_only / sentinel rows excluded).
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    filtered = _filter_labels(labels, decision_filter)
    drawable = [label for label in filtered if _include_in_rerun(label)]

    frames: dict[tuple[str, int], list[PseudoLabelRecord]] = defaultdict(list)
    for label in drawable:
        frames[(label.scene_id, label.timestamp_us)].append(label)

    recording = RecordingStream(application_id)
    recording.save(output)

    for (scene_id, timestamp_us), frame_labels in sorted(frames.items()):
        recording.set_time("timestamp", timestamp=timestamp_us / 1e6)
        centers = [label.center_xyz for label in frame_labels]
        half_sizes = [
            [dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0]
            for dims in (label.dimensions_whl for label in frame_labels)
        ]
        rotations = [
            rr.RotationAxisAngle(axis=[0.0, 0.0, 1.0], angle=rr.datatypes.Angle(rad=label.yaw))
            for label in frame_labels
        ]
        labels_text = [
            f"{label.class_name} ({label.fusion_type}, trust={label.trust_score:.2f})"
            for label in frame_labels
        ]
        recording.log(
            f"scenes/{scene_id}/boxes",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                rotations=rotations,
                labels=labels_text,
            ),
        )

    recording.flush()
    recording.disconnect()

    return len(drawable)
