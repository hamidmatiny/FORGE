"""Generate synthetic mini_lake fixture for CI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from forge.schemas.frames import FrameRecord, FramesTable

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mini_lake"
OUTPUT_PATH = FIXTURE_DIR / "frames.parquet"

SYNTHETIC_FRAMES: list[FrameRecord] = [
    FrameRecord(
        frame_id="frame-001",
        scene_id="scene-alpha",
        timestamp_us=1_700_000_000_000_000,
        sensor_id="CAM_FRONT",
        dataset_split="train",
        data_path="samples/CAM_FRONT/scene-alpha_00.jpg",
        ingested_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    ),
    FrameRecord(
        frame_id="frame-002",
        scene_id="scene-alpha",
        timestamp_us=1_700_000_050_000_000,
        sensor_id="LIDAR_TOP",
        dataset_split="train",
        data_path="samples/LIDAR_TOP/scene-alpha_00.pcd.bin",
        ingested_at=datetime(2024, 1, 15, 12, 0, 0, 500000, tzinfo=UTC),
    ),
    FrameRecord(
        frame_id="frame-003",
        scene_id="scene-beta",
        timestamp_us=1_700_000_100_000_000,
        sensor_id="CAM_BACK",
        dataset_split="val",
        data_path="samples/CAM_BACK/scene-beta_00.jpg",
        ingested_at=datetime(2024, 1, 15, 12, 0, 1, tzinfo=UTC),
    ),
]


def main() -> None:
    """Write the synthetic frames fixture to tests/fixtures/mini_lake/."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FramesTable.write_parquet(SYNTHETIC_FRAMES, str(OUTPUT_PATH))
    print(f"Wrote {len(SYNTHETIC_FRAMES)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
