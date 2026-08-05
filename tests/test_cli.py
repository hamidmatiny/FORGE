"""Tests for the FORGE CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()


def test_help_lists_all_stage_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output
    for command in (
        "ingest",
        "detect2d",
        "detect3d",
        "track",
        "fuse",
        "label",
        "evaluate",
        "curate",
        "visualize",
    ):
        assert command in output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "forge 0.1.0" in result.output


def test_ingest_requires_input_dir() -> None:
    result = runner.invoke(app, ["ingest", "--local"])
    assert result.exit_code != 0
    assert "input-dir" in result.output.lower() or "input_dir" in result.output.lower()


def test_ingest_requires_local_flag() -> None:
    result = runner.invoke(app, ["ingest", "--input-dir", "tests/fixtures/nuscenes_mini_synthetic"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_ingest_succeeds_against_synthetic_fixture(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ingest",
            "--input-dir",
            "tests/fixtures/nuscenes_mini_synthetic",
            "--local",
        ],
        env={"FORGE_DATA_LAKE_ROOT": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "wrote 5 frames" in result.output


def test_detect2d_requires_local_flag() -> None:
    result = runner.invoke(app, ["detect2d"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_detect2d_infer_requires_images_root() -> None:
    result = runner.invoke(app, ["detect2d", "--mode", "infer", "--local"])
    assert result.exit_code == 1
    assert "images-root" in result.output.lower()


def test_detect2d_unknown_mode_errors() -> None:
    result = runner.invoke(app, ["detect2d", "--mode", "bogus", "--local"])
    assert result.exit_code == 1
    assert "Unknown --mode" in result.output


def test_detect3d_requires_local_flag() -> None:
    result = runner.invoke(app, ["detect3d"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_detect3d_infer_requires_pointcloud_root() -> None:
    result = runner.invoke(app, ["detect3d", "--mode", "infer", "--local"])
    assert result.exit_code == 1
    assert "pointcloud-root" in result.output.lower()


def test_detect3d_unknown_mode_errors() -> None:
    result = runner.invoke(app, ["detect3d", "--mode", "bogus", "--local"])
    assert result.exit_code == 1
    assert "Unknown --mode" in result.output


def test_track_requires_local_flag() -> None:
    result = runner.invoke(app, ["track"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_track_requires_frames_lake(tmp_path: Path) -> None:
    result = runner.invoke(app, ["track", "--local"], env={"FORGE_DATA_LAKE_ROOT": str(tmp_path)})
    assert result.exit_code == 1
    assert "forge ingest" in result.output


def test_fuse_requires_local_flag() -> None:
    result = runner.invoke(app, ["fuse"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_fuse_requires_frames_lake(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fuse", "--local"], env={"FORGE_DATA_LAKE_ROOT": str(tmp_path)})
    assert result.exit_code == 1
    assert "forge ingest" in result.output


def test_label_requires_local_flag() -> None:
    result = runner.invoke(app, ["label"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_label_requires_fused_objects_lake(tmp_path: Path) -> None:
    result = runner.invoke(app, ["label", "--local"], env={"FORGE_DATA_LAKE_ROOT": str(tmp_path)})
    assert result.exit_code == 1
    assert "forge fuse" in result.output


def test_evaluate_requires_local_flag() -> None:
    result = runner.invoke(app, ["evaluate", "--gt-input-dir", "some/path"])
    assert result.exit_code == 1
    assert "Phase 9" in result.output


def test_evaluate_requires_pseudo_labels_lake(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["evaluate", "--gt-input-dir", "some/path", "--local"],
        env={"FORGE_DATA_LAKE_ROOT": str(tmp_path)},
    )
    assert result.exit_code == 1
    assert "forge label" in result.output
