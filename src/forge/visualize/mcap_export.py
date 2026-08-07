"""Write pseudo-labels to an MCAP file with plain JSON messages (not Foxglove native schemas)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from mcap.writer import Writer

from forge.schemas import PseudoLabelRecord

MCAP_SCHEMA_NAME = "forge.pseudo_labels.v1"
MCAP_CHANNEL_TOPIC = "/forge/pseudo_labels"


def _filter_labels(
    labels: list[PseudoLabelRecord],
    decision_filter: str,
) -> list[PseudoLabelRecord]:
    if decision_filter == "all":
        return labels
    return [label for label in labels if label.decision == decision_filter]


def _label_to_dict(label: PseudoLabelRecord) -> dict[str, object]:
    return {
        "pseudo_label_id": label.pseudo_label_id,
        "fusion_id": label.fusion_id,
        "scene_id": label.scene_id,
        "timestamp_us": label.timestamp_us,
        "fusion_type": label.fusion_type,
        "class_id": label.class_id,
        "class_name": label.class_name,
        "bbox_xyxy": label.bbox_xyxy,
        "center_xyz": label.center_xyz,
        "dimensions_whl": label.dimensions_whl,
        "yaw": label.yaw,
        "trust_score": label.trust_score,
        "decision": label.decision,
        "review_priority": label.review_priority,
        "labeler_version": label.labeler_version,
    }


def build_mcap_recording(
    labels: list[PseudoLabelRecord],
    path: Path | str,
    *,
    decision_filter: str = "auto_accept",
) -> int:
    """Write one MCAP message per (scene_id, timestamp_us) with all labels in that frame.

    Returns the total number of pseudo-label rows written (includes camera_only rows).
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    filtered = _filter_labels(labels, decision_filter)
    frames: dict[tuple[str, int], list[PseudoLabelRecord]] = defaultdict(list)
    for label in filtered:
        frames[(label.scene_id, label.timestamp_us)].append(label)

    schema_json = json.dumps(
        {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string"},
                "timestamp_us": {"type": "integer"},
                "objects": {"type": "array"},
            },
        }
    )

    with output.open("wb") as file_handle:
        writer = Writer(file_handle)
        writer.start()
        schema_id = writer.register_schema(
            name=MCAP_SCHEMA_NAME,
            encoding="jsonschema",
            data=schema_json.encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic=MCAP_CHANNEL_TOPIC,
            message_encoding="json",
            schema_id=schema_id,
        )

        for (scene_id, timestamp_us), frame_labels in sorted(frames.items()):
            payload = {
                "scene_id": scene_id,
                "timestamp_us": timestamp_us,
                "objects": [_label_to_dict(label) for label in frame_labels],
            }
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_us * 1000,
                publish_time=timestamp_us * 1000,
                data=data,
            )
        writer.finish()

    return len(filtered)
