"""S3-upload-triggered Lambda: validate nuScenes file layout, notify SQS + EventBridge.

Deployed via Terraform (see ../../terraform/lambda.tf), triggered on
``s3:ObjectCreated:*`` for the raw-data bucket. Deliberately lightweight —
Lambda has execution time/memory limits unsuitable for the actual ML
pipeline work (that's what Ray/ECS are for, per Phase 9's other half).
This function's job is: does this uploaded file look like a real
nuScenes file, and if so, tell downstream consumers about it — via SQS
(a plain queue a simple consumer could poll, on every valid upload) and,
once a dataset looks minimally complete (see
``_check_and_record_completeness``), via EventBridge (which triggers the
Step Functions pipeline orchestration, see ../../terraform/step_functions.tf
and eventbridge.tf).

Completeness tracking uses a DynamoDB table (one item per
``dataset_root``, a string-set attribute of file categories seen so far)
rather than triggering the pipeline on every single upload — see
``_check_and_record_completeness``'s docstring for the heuristic and its
real limitations.

Not part of the ``forge`` package — Lambda deployment packages are
typically standalone, zip-deployed modules, not full installable
packages, so this lives outside ``src/forge/`` deliberately (matches how
this would really be packaged for deployment).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import boto3

# nuScenes-devkit layout (see forge.ingest.nuscenes for the same convention
# on the ingest side): "<version>/*.json" metadata tables, or raw sensor
# files under "samples/**" / "sweeps/**". Both may sit under an arbitrary
# prefix (e.g. a real bucket might use "fleet-a/2026-01/v1.0-mini/...") --
# the (?P<root>.*?) group captures whatever that prefix is.
_METADATA_TABLE_RE = re.compile(r"^(?P<root>.*?)v\d+\.\d+(-\w+)?/[\w_]+\.json$")
_SENSOR_FILE_RE = re.compile(r"^(?P<root>.*?)(samples|sweeps)/[\w_]+/.+$")

_EVENT_SOURCE = "forge.ingest"
_EVENT_DETAIL_TYPE = "IngestUploadValidated"

# A dataset is considered "minimally complete" once at least one file of
# each category has been seen for its dataset_root -- see
# _check_and_record_completeness's docstring for what this does and
# doesn't guarantee.
_REQUIRED_CATEGORIES = frozenset({"metadata_table", "sensor_file"})


@dataclass(frozen=True)
class IngestNotification:
    """The message published to SQS and EventBridge for a validated upload."""

    bucket: str
    key: str
    size_bytes: int
    event_time: str
    file_category: str  # "metadata_table" | "sensor_file"
    dataset_root: str  # best-effort: the key prefix identifying the dataset


def validate_key(key: str) -> bool:
    """True if `key` matches the expected nuScenes-devkit layout."""
    return bool(_METADATA_TABLE_RE.match(key) or _SENSOR_FILE_RE.match(key))


def _infer_dataset_root(key: str) -> str:
    """Best-effort dataset root: whatever prefix precedes the recognized layout piece.

    A real multi-dataset bucket might have several nuScenes drops under
    different prefixes (e.g. ``fleet-a/2026-01/v1.0-mini/scene.json``) --
    this returns that prefix so a downstream consumer knows which drop a
    file belongs to. Empty string means the dataset root is the bucket
    itself (the standard layout starts right at the key root).
    """
    metadata_match = _METADATA_TABLE_RE.match(key)
    if metadata_match:
        return metadata_match.group("root")
    sensor_match = _SENSOR_FILE_RE.match(key)
    if sensor_match:
        return sensor_match.group("root")
    return ""


def build_notification(
    bucket: str, key: str, size_bytes: int, event_time: str
) -> IngestNotification:
    """Build the SQS message payload for a validated upload.

    Raises:
        ValueError: If `key` doesn't match the expected layout -- callers
            should check `validate_key` first.
    """
    if not validate_key(key):
        raise ValueError(f"'{key}' doesn't match the expected nuScenes layout.")

    file_category = "metadata_table" if _METADATA_TABLE_RE.match(key) else "sensor_file"
    return IngestNotification(
        bucket=bucket,
        key=key,
        size_bytes=size_bytes,
        event_time=event_time,
        file_category=file_category,
        dataset_root=_infer_dataset_root(key),
    )


def _check_and_record_completeness(
    dynamodb_client: Any,  # noqa: ANN401 - untyped boto3 client, no stubs installed
    table_name: str,
    dataset_root: str,
    file_category: str,
) -> bool:
    """Record `file_category` as seen for `dataset_root`; return True if this

    specific update is what just made the dataset newly "complete".

    Heuristic, not a guarantee: "complete" means at least one
    metadata_table file and at least one sensor_file have been seen for
    this dataset_root -- NOT that every expected file has arrived. A real
    nuScenes drop has many metadata tables and many sensor files; this
    doesn't count or verify any of them individually, it only checks that
    both *categories* have shown up at least once. Chosen over a fully
    rigorous per-file completeness design (e.g. an expected manifest with
    a count to match) because that's real, unbuildable-without-guessing
    scope for this pass -- see DECISIONS.md and KNOWN_GAPS.md.

    Returns False (and does NOT re-publish) on every upload after the
    dataset first became complete, so one dataset triggers the pipeline
    at most once from this path.
    """
    existing = dynamodb_client.get_item(
        TableName=table_name, Key={"dataset_root": {"S": dataset_root}}
    )
    previously_seen: frozenset[str] = frozenset(
        existing.get("Item", {}).get("categories_seen", {}).get("SS", [])
    )
    was_complete = _REQUIRED_CATEGORIES.issubset(previously_seen)

    dynamodb_client.update_item(
        TableName=table_name,
        Key={"dataset_root": {"S": dataset_root}},
        UpdateExpression="ADD categories_seen :cat",
        ExpressionAttributeValues={":cat": {"SS": [file_category]}},
    )

    now_seen = previously_seen | {file_category}
    is_complete_now = _REQUIRED_CATEGORIES.issubset(now_seen)

    return is_complete_now and not was_complete


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, int]:
    """Entry point: validate each S3 record, publish to SQS always, EventBridge once-complete.

    SQS gets every valid upload (a simple "stuff arrived" queue). EventBridge
    — which triggers the Step Functions pipeline — only gets published the
    first time a dataset_root's completeness heuristic is satisfied (see
    `_check_and_record_completeness`), not on every single upload.

    Returns:
        Summary counts: ``{"processed": N, "published": M, "triggered": T, "skipped": K}``.
        `published` counts valid uploads (always sent to SQS); `triggered`
        counts how many of those also started a pipeline run via EventBridge.
    """
    queue_url = os.environ["INGEST_QUEUE_URL"]
    event_bus_name = os.environ["EVENT_BUS_NAME"]
    completeness_table_name = os.environ["COMPLETENESS_TABLE_NAME"]
    sqs = boto3.client("sqs")
    events_client = boto3.client("events")
    dynamodb_client = boto3.client("dynamodb")

    processed = 0
    published = 0
    triggered = 0
    skipped = 0

    for record in event.get("Records", []):
        processed += 1
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name", "")
        key = s3_info.get("object", {}).get("key", "")
        size = int(s3_info.get("object", {}).get("size", 0))
        event_time = record.get("eventTime", "")

        if not validate_key(key):
            skipped += 1
            continue

        notification = build_notification(bucket, key, size, event_time)
        payload = json.dumps(asdict(notification))

        sqs.send_message(QueueUrl=queue_url, MessageBody=payload)
        published += 1

        just_completed = _check_and_record_completeness(
            dynamodb_client,
            completeness_table_name,
            notification.dataset_root,
            notification.file_category,
        )
        if just_completed:
            events_client.put_events(
                Entries=[
                    {
                        "Source": _EVENT_SOURCE,
                        "DetailType": _EVENT_DETAIL_TYPE,
                        "Detail": payload,
                        "EventBusName": event_bus_name,
                    }
                ]
            )
            triggered += 1

    return {
        "processed": processed,
        "published": published,
        "triggered": triggered,
        "skipped": skipped,
    }
