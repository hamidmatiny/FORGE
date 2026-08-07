"""Tests for infra/lambda/ingest_trigger/handler.py.

Pure logic (validate_key, build_notification) needs no mocking. The
handler itself mocks boto3's SQS, EventBridge, and DynamoDB clients --
these tests never touch real AWS, matching the cost-safety policy every
other phase follows. DynamoDB is mocked with a small in-memory fake that
actually implements get_item/update_item (not a bare MagicMock), since
the completeness-gating logic genuinely depends on realistic behavior
across calls, not just that the right methods got called.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("boto3")

from handler import (  # noqa: E402
    IngestNotification,
    _check_and_record_completeness,
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


# --- _check_and_record_completeness --------------------------------------


class _FakeDynamoDB:
    """A minimal, real (not mocked) in-memory implementation of get_item/update_item

    scoped to exactly what _check_and_record_completeness calls. Genuinely
    tracks state across calls, unlike a bare MagicMock -- this logic's
    correctness depends on that.
    """

    def __init__(self) -> None:
        self.store: dict[str, set[str]] = {}

    def get_item(self, TableName: str, Key: dict[str, Any]) -> dict[str, Any]:
        root = Key["dataset_root"]["S"]
        if root not in self.store:
            return {}
        return {"Item": {"categories_seen": {"SS": list(self.store[root])}}}

    def update_item(
        self,
        TableName: str,
        Key: dict[str, Any],
        UpdateExpression: str,
        ExpressionAttributeValues: dict[str, Any],
    ) -> None:
        root = Key["dataset_root"]["S"]
        categories = ExpressionAttributeValues[":cat"]["SS"]
        self.store.setdefault(root, set()).update(categories)


def test_completeness_first_category_never_triggers() -> None:
    db = _FakeDynamoDB()
    assert _check_and_record_completeness(db, "t", "root-a", "metadata_table") is False


def test_completeness_repeated_category_never_triggers() -> None:
    db = _FakeDynamoDB()
    _check_and_record_completeness(db, "t", "root-a", "metadata_table")
    assert _check_and_record_completeness(db, "t", "root-a", "metadata_table") is False


def test_completeness_second_distinct_category_triggers_exactly_once() -> None:
    db = _FakeDynamoDB()
    _check_and_record_completeness(db, "t", "root-a", "metadata_table")
    assert _check_and_record_completeness(db, "t", "root-a", "sensor_file") is True


def test_completeness_already_complete_dataset_never_retriggers() -> None:
    db = _FakeDynamoDB()
    _check_and_record_completeness(db, "t", "root-a", "metadata_table")
    _check_and_record_completeness(db, "t", "root-a", "sensor_file")  # triggers here
    assert _check_and_record_completeness(db, "t", "root-a", "sensor_file") is False
    assert _check_and_record_completeness(db, "t", "root-a", "metadata_table") is False


def test_completeness_different_dataset_roots_are_isolated() -> None:
    db = _FakeDynamoDB()
    _check_and_record_completeness(db, "t", "root-a", "metadata_table")
    _check_and_record_completeness(db, "t", "root-a", "sensor_file")  # root-a complete
    # root-b has seen nothing yet -- must not inherit root-a's completeness.
    assert _check_and_record_completeness(db, "t", "root-b", "sensor_file") is False


# --- lambda_handler -----------------------------------------------------

_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123/queue"
_EVENT_BUS_NAME = "forge-events-dev"
_COMPLETENESS_TABLE_NAME = "forge-dataset-completeness-dev"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_QUEUE_URL", _QUEUE_URL)
    monkeypatch.setenv("EVENT_BUS_NAME", _EVENT_BUS_NAME)
    monkeypatch.setenv("COMPLETENESS_TABLE_NAME", _COMPLETENESS_TABLE_NAME)


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


def _mock_boto3_clients() -> tuple[Any, MagicMock, MagicMock, _FakeDynamoDB]:
    """Return (patched boto3.client factory, mock sqs client, mock events client, fake dynamodb)."""
    mock_sqs = MagicMock()
    mock_events = MagicMock()
    fake_dynamodb = _FakeDynamoDB()

    def factory(service_name: str) -> Any:
        return {"sqs": mock_sqs, "events": mock_events, "dynamodb": fake_dynamodb}[service_name]

    return factory, mock_sqs, mock_events, fake_dynamodb


def test_lambda_handler_publishes_every_valid_upload_to_sqs_regardless_of_completeness() -> None:
    """SQS is the 'everything' queue -- gets every valid upload, not gated by completeness."""
    event = _s3_event(
        [
            ("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100),  # metadata_table
            ("2026-08-06T00:00:01Z", "samples/CAM_FRONT/x.jpg", 200),  # sensor_file -> completes
            ("2026-08-06T00:00:02Z", "garbage.txt", 50),  # invalid, skipped
        ]
    )
    factory, mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        result = lambda_handler(event, None)

    assert result == {"processed": 3, "published": 2, "triggered": 1, "skipped": 1}
    assert mock_sqs.send_message.call_count == 2
    # Only the SECOND valid upload completes the dataset -- exactly one trigger.
    assert mock_events.put_events.call_count == 1


def test_lambda_handler_single_category_upload_never_triggers_eventbridge() -> None:
    """A lone metadata_table upload (no sensor_file yet) must not start a pipeline run."""
    event = _s3_event([("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100)])
    factory, mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        result = lambda_handler(event, None)

    assert result == {"processed": 1, "published": 1, "triggered": 0, "skipped": 0}
    mock_sqs.send_message.assert_called_once()
    mock_events.put_events.assert_not_called()


def test_lambda_handler_sends_correct_sqs_message_body() -> None:
    event = _s3_event([("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100)])
    factory, mock_sqs, _mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        lambda_handler(event, None)

    call = mock_sqs.send_message.call_args
    assert call.kwargs["QueueUrl"] == _QUEUE_URL
    body = json.loads(call.kwargs["MessageBody"])
    assert body["bucket"] == "b"
    assert body["key"] == "v1.0-mini/scene.json"
    assert body["file_category"] == "metadata_table"


def test_lambda_handler_sends_correct_eventbridge_entry_on_completion() -> None:
    """EventBridge publication is what triggers Step Functions -- verify its shape

    on the upload that actually completes the dataset (the second, distinct
    category -- a single-category upload never reaches this path, see the
    dedicated test above).
    """
    event = _s3_event(
        [
            ("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100),
            ("2026-08-06T00:00:01Z", "samples/CAM_FRONT/x.jpg", 200),
        ]
    )
    factory, _mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        lambda_handler(event, None)

    call = mock_events.put_events.call_args
    entries = call.kwargs["Entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["Source"] == "forge.ingest"
    assert entry["DetailType"] == "IngestUploadValidated"
    assert entry["EventBusName"] == _EVENT_BUS_NAME
    detail = json.loads(entry["Detail"])
    assert detail["key"] == "samples/CAM_FRONT/x.jpg"  # the completing upload


def test_lambda_handler_sqs_and_eventbridge_get_identical_payloads_for_the_triggering_upload() -> (
    None
):
    event = _s3_event(
        [
            ("2026-08-06T00:00:00Z", "v1.0-mini/scene.json", 100),
            ("2026-08-06T00:00:01Z", "samples/CAM_FRONT/x.jpg", 200),
        ]
    )
    factory, mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        lambda_handler(event, None)

    last_sqs_body = mock_sqs.send_message.call_args.kwargs["MessageBody"]
    eventbridge_detail = mock_events.put_events.call_args.kwargs["Entries"][0]["Detail"]
    assert json.loads(last_sqs_body) == json.loads(eventbridge_detail)


def test_lambda_handler_empty_records() -> None:
    factory, mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        result = lambda_handler({"Records": []}, None)

    assert result == {"processed": 0, "published": 0, "triggered": 0, "skipped": 0}
    mock_sqs.send_message.assert_not_called()
    mock_events.put_events.assert_not_called()


def test_lambda_handler_all_invalid_never_publishes() -> None:
    event = _s3_event([("2026-08-06T00:00:00Z", "garbage.txt", 10)])
    factory, mock_sqs, mock_events, _fake_dynamodb = _mock_boto3_clients()

    with patch("handler.boto3.client", side_effect=factory):
        result = lambda_handler(event, None)

    assert result == {"processed": 1, "published": 0, "triggered": 0, "skipped": 1}
    mock_sqs.send_message.assert_not_called()
    mock_events.put_events.assert_not_called()
