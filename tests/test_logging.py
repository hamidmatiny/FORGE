"""Tests for structured logging."""

from __future__ import annotations

import json

import pytest

from forge.logging import configure_logging, get_logger, log_cli_invocation


def test_configure_logging_and_get_logger() -> None:
    configure_logging("DEBUG")
    logger = get_logger("forge.test")
    assert logger is not None


def test_log_cli_invocation_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    log_cli_invocation("ingest", {"local": True}, log_level="INFO")
    captured = capsys.readouterr()
    assert captured.err
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["event"] == "cli_invocation"
    assert payload["command"] == "ingest"
    assert payload["args"] == {"local": True}
    assert payload["version"] == "0.1.0"
    assert "git_sha" in payload
