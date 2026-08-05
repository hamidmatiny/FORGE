"""Tests for schema base machinery and frames table."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from forge.schemas.base import SchemaVersion
from forge.schemas.frames import FrameRecord, FramesTable

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mini_lake" / "frames.parquet"


def test_schema_version_str() -> None:
    assert str(SchemaVersion(major=1, minor=0)) == "v1.0"


def test_frames_arrow_schema_field_count() -> None:
    schema = FramesTable.arrow_schema()
    assert len(schema) == 7
    assert schema.field("timestamp_us").type == pa.int64()


def test_round_trip_pydantic_arrow_pydantic() -> None:
    records = [
        FrameRecord(
            frame_id="f1",
            scene_id="s1",
            timestamp_us=1000,
            sensor_id="CAM_FRONT",
            dataset_split="train",
            data_path="samples/CAM_FRONT/f1.jpg",
            ingested_at=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    ]
    table = FramesTable.to_arrow(records)
    round_tripped = FramesTable.from_arrow(table)
    assert round_tripped == records


def test_round_trip_parquet(tmp_path: Path) -> None:
    records = [
        FrameRecord(
            frame_id="f1",
            scene_id="s1",
            timestamp_us=1000,
            sensor_id="CAM_FRONT",
            dataset_split="train",
            data_path="samples/CAM_FRONT/f1.jpg",
            ingested_at=datetime(2024, 6, 1, tzinfo=UTC),
        ),
        FrameRecord(
            frame_id="f2",
            scene_id="s1",
            timestamp_us=2000,
            sensor_id="LIDAR_TOP",
            dataset_split="val",
            data_path="samples/LIDAR_TOP/f2.pcd.bin",
            ingested_at=datetime(2024, 6, 1, 0, 0, 1, tzinfo=UTC),
        ),
    ]
    path = tmp_path / "frames.parquet"
    FramesTable.write_parquet(records, str(path))
    loaded = FramesTable.read_parquet(str(path))
    assert loaded == records


def test_committed_fixture_has_three_rows() -> None:
    assert FIXTURE_PATH.exists(), "Run scripts/make_fixture.py to generate fixture"
    records = FramesTable.read_parquet(str(FIXTURE_PATH))
    assert len(records) == 3
    assert records[0].frame_id == "frame-001"


def test_empty_records_produce_valid_arrow_table() -> None:
    table = FramesTable.to_arrow([])
    assert table.num_rows == 0
    assert table.schema.equals(FramesTable.arrow_schema())


@hyp_settings(max_examples=50, deadline=None)
@given(
    frame_id=st.text(min_size=1, max_size=64),
    scene_id=st.text(min_size=1, max_size=64),
    timestamp_us=st.integers(min_value=0, max_value=2**62),
    sensor_id=st.sampled_from(["CAM_FRONT", "CAM_BACK", "LIDAR_TOP"]),
    dataset_split=st.sampled_from(["train", "val", "test", "unknown"]),
)
def test_frame_record_round_trip_properties(
    frame_id: str,
    scene_id: str,
    timestamp_us: int,
    sensor_id: str,
    dataset_split: str,
) -> None:
    ingested_at = datetime(2024, 1, 1, tzinfo=UTC)
    record = FrameRecord(
        frame_id=frame_id,
        scene_id=scene_id,
        timestamp_us=timestamp_us,
        sensor_id=sensor_id,
        dataset_split=dataset_split,
        ingested_at=ingested_at,
    )
    round_tripped = FramesTable.from_arrow(FramesTable.to_arrow([record]))
    assert round_tripped[0] == record
