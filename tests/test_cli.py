"""Tests for the FORGE CLI."""

from __future__ import annotations

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


def test_ingest_exits_nonzero_with_message() -> None:
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "not implemented until Phase 1" in result.output
    assert "KNOWN_GAPS.md" in result.output


def test_detect2d_exits_nonzero() -> None:
    result = runner.invoke(app, ["detect2d"])
    assert result.exit_code == 1
    assert "Phase 2" in result.output


def test_local_flag_accepted() -> None:
    result = runner.invoke(app, ["ingest", "--local"])
    assert result.exit_code == 1
