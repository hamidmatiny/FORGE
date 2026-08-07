"""Tests for forge.visualize exports."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from mcap.reader import make_reader

from forge.schemas import PseudoLabelRecord, PseudoLabelsTable
from forge.visualize.mcap_export import MCAP_CHANNEL_TOPIC, build_mcap_recording
from forge.visualize.rerun_export import build_rerun_recording

pytest.importorskip("rerun")
pytest.importorskip("mcap")


def _label(
    pseudo_label_id: str,
    *,
    fusion_type: str = "matched",
    center_xyz: list[float] | None = None,
    decision: str = "auto_accept",
    timestamp_us: int = 1_000,
    scene_id: str = "scene-a",
) -> PseudoLabelRecord:
    if center_xyz is None:
        center_xyz = [10.0, 0.0, 0.0]
    return PseudoLabelRecord(
        pseudo_label_id=pseudo_label_id,
        fusion_id=f"f-{pseudo_label_id}",
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        fusion_type=fusion_type,
        class_id=1,
        class_name="vehicle",
        bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        center_xyz=center_xyz,
        dimensions_whl=[4.0, 2.0, 1.5],
        yaw=0.1,
        trust_score=0.9,
        decision=decision,
        review_priority=0.2,
        labeler_version="test-v1",
    )


def test_rerun_export_skips_camera_only_sentinel(tmp_path: Path) -> None:
    labels = [
        _label("grounded"),
        _label(
            "cam-only",
            fusion_type="camera_only",
            center_xyz=[0.0, 0.0, 0.0],
        ),
    ]
    path = tmp_path / "out.rrd"
    count = build_rerun_recording(labels, path, decision_filter="all")
    assert count == 1
    assert path.exists()


def test_mcap_export_includes_camera_only(tmp_path: Path) -> None:
    labels = [
        _label("grounded"),
        _label(
            "cam-only",
            fusion_type="camera_only",
            center_xyz=[0.0, 0.0, 0.0],
        ),
    ]
    path = tmp_path / "out.mcap"
    count = build_mcap_recording(labels, path, decision_filter="all")
    assert count == 2
    with path.open("rb") as handle:
        reader = make_reader(handle)
        messages = list(reader.iter_messages(topics=[MCAP_CHANNEL_TOPIC]))
    assert len(messages) == 1
    _schema, _channel, message = messages[0]
    payload = json.loads(message.data)
    assert len(payload["objects"]) == 2


def test_decision_filter_applied(tmp_path: Path) -> None:
    labels = [
        _label("keep", decision="auto_accept"),
        _label("drop", decision="rejected"),
    ]
    path = tmp_path / "out.mcap"
    count = build_mcap_recording(labels, path, decision_filter="auto_accept")
    assert count == 1


def test_rerun_export_file_verifies_with_rerun_cli(tmp_path: Path) -> None:
    rerun_bin = shutil.which("rerun")
    if rerun_bin is None:
        pytest.skip("rerun CLI not on PATH")

    labels = [_label("a"), _label("b", timestamp_us=2_000, center_xyz=[5.0, 1.0, 0.0])]
    path = tmp_path / "verify.rrd"
    build_rerun_recording(labels, path, decision_filter="all")

    result = subprocess.run(
        [rerun_bin, "rrd", "verify", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_build_rerun_recording_empty_labels(tmp_path: Path) -> None:
    path = tmp_path / "empty.rrd"
    count = build_rerun_recording([], path, decision_filter="all")
    assert count == 0
    assert path.exists()


def test_pseudo_labels_table_round_trip_fixture() -> None:
    """Sanity: visualize tests use the same schema as the lake."""
    record = _label("x")
    table = PseudoLabelsTable.to_arrow([record])
    round_tripped = PseudoLabelsTable.from_arrow(table)
    assert round_tripped[0].pseudo_label_id == record.pseudo_label_id
    assert round_tripped[0].fusion_type == record.fusion_type


def test_mcap_sorted_frame_messages(tmp_path: Path) -> None:
    labels = [
        _label("later", timestamp_us=3_000),
        _label("earlier", timestamp_us=1_000, center_xyz=[1.0, 0.0, 0.0]),
    ]
    path = tmp_path / "frames.mcap"
    build_mcap_recording(labels, path, decision_filter="all")
    with path.open("rb") as handle:
        reader = make_reader(handle)
        timestamps = []
        for _schema, channel, message in reader.iter_messages(topics=[MCAP_CHANNEL_TOPIC]):
            if channel.topic != MCAP_CHANNEL_TOPIC:
                continue
            payload = json.loads(message.data)
            timestamps.append(payload["timestamp_us"])
    assert timestamps == [1_000, 3_000]
