"""Tests for infra/lambda/ingest_trigger/handler.py.

Pure logic (validate_key, build_notification) needs no mocking. The
handler itself mocks boto3's SQS client -- these tests never touch real
AWS, matching the cost-safety policy every other phase follows.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("boto3")

from handler import (  # noqa: E402
    IngestNotification,
    build_notification,
    lambda_handler,
    validate_key,
)

# --- validate_key -----------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "v1.0-mini/scene.json",
        "v1.0-trainval/sample_data.json",
        "samples/CAM_FRONT/scene0001_00.jpg",
        "sweeps/LIDAR_TOP/scene0001_00b.pcd.bin",
        "fleet-a/2026-01/v1.0-mini/scene.json",
        "fleet-a/2026-01/samples/CAM_FRONT/x.jpg",
    ],
)
def test_validate_key_accepts_nuscenes_layout(key: str) -> None:
    assert validate_key(key) is True


@pytest.mark.parametrize(
    "key",
    [
        "random/garbage.txt",
        "v1.0-mini/scene.json.bak",
        "notes.md",
        "samples/CAM_FRONT",  # missing filename
        "",
    ],
)
def test_validate_key_rejects_non_nuscenes_paths(key: str) -> None:
    assert validate_key(key) is False


# --- build_notification --------------------------------------------------


def test_build_notification_metadata_table_at_bucket_root() -> None:
    notification = build_notification("b", "v1.0-mini/scene.json", 1024, "2026-08-06T00:00:00Z")
    assert notification == IngestNotification(
        bucket="b",
        key="v1.0-mini/scene.json",
        size_bytes=1024,
        event_time="2026-08-06T00:00:00Z",
        file_category="metadata_table",
        dataset_root="",
    )


def test_build_notification_sensor_file_under_arbitrary_prefix() -> None:
    notification = build_notification(
        "b", "fleet-a/2026-01/samples/CAM_FRONT/x.jpg", 2048, "2026-08-06T00:00:01Z"
    )
    assert notification.file_category == "sensor_file"
    assert notification.dataset_root == "fleet-a/2026-01/"


def test_build_notification_rejects_invalid_key() -> None:
    with pytest.raises(ValueError, match="doesn't match"):
        build_notification("b", "garbage.txt", 10, "2026-08-06T00:00:00Z")


# --- lambda_handler -----------------------------------------------------


@pytest.fixture(autouse=True)
def _queue_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123/queue")


def _s3_event(records: list[tuple[str, str, int]]) -> dict:
    return {
        "Records": [
            {
                "eventTime": event_time,
                "s3": {"bucket": {"name": "b"}, "object": {"key": key, "size": size}},
            }
            for event_time, key, size in records
        ]
    }


def test_lambda_handler_publishes_valid_uploads_and_skips_invalid_ones() -> None:
    event = _s3_event(
        [
            ("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100),
            ("2026-08-06T00:00:01Z", "samples/CAM_FRONT/x.jpg", 200),
            ("2026-08-06T00:00:02Z", "garbage.txt", 50),
        ]
    )

    with patch("handler.boto3.client") as mock_client_factory:
        mock_sqs = MagicMock()
        mock_client_factory.return_value = mock_sqs
        result = lambda_handler(event, None)

    assert result == {"processed": 3, "published": 2, "skipped": 1}
    assert mock_sqs.send_message.call_count == 2


def test_lambda_handler_sends_correct_message_body() -> None:
    event = _s3_event([("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100)])

    with patch("handler.boto3.client") as mock_client_factory:
        mock_sqs = MagicMock()
        mock_client_factory.return_value = mock_sqs
        lambda_handler(event, None)

    call = mock_sqs.send_message.call_args
    assert call.kwargs["QueueUrl"] == os.environ["INGEST_QUEUE_URL"]
    body = json.loads(call.kwargs["MessageBody"])
    assert body["bucket"] == "b"
    assert body["key"] == "v1.0-mini/scene.json"
    assert body["file_category"] == "metadata_table"


def test_lambda_handler_empty_records() -> None:
    with patch("handler.boto3.client") as mock_client_factory:
        mock_sqs = MagicMock()
        mock_client_factory.return_value = mock_sqs
        result = lambda_handler({"Records": []}, None)

    assert result == {"processed": 0, "published": 0, "skipped": 0}
    mock_sqs.send_message.assert_not_called()


def test_lambda_handler_all_invalid_never_calls_sqs() -> None:
    event = _s3_event([("2026-08-06T00:00:00Z", "garbage.txt", 10)])

    with patch("handler.boto3.client") as mock_client_factory:
        mock_sqs = MagicMock()
        mock_client_factory.return_value = mock_sqs
        result = lambda_handler(event, None)

    assert result == {"processed": 1, "published": 0, "skipped": 1}
    mock_sqs.send_message.assert_not_called()
