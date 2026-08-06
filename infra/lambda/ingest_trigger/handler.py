"""S3-upload-triggered Lambda: validate nuScenes file layout, notify SQS.

Deployed via Terraform (see ../../terraform/lambda.tf), triggered on
``s3:ObjectCreated:*`` for the raw-data bucket. Deliberately lightweight —
Lambda has execution time/memory limits unsuitable for the actual ML
pipeline work (that's what Ray/ECS are for, per Phase 9's other half).
This function's only job is: does this uploaded file look like a real
nuScenes file, and if so, tell a downstream queue about it.

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


@dataclass(frozen=True)
class IngestNotification:
    """The message published to SQS for a validated upload."""

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


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, int]:
    """Entry point: validate each S3 record in the event, publish valid ones to SQS.

    Returns:
        Summary counts: ``{"processed": N, "published": M, "skipped": K}``.
    """
    queue_url = os.environ["INGEST_QUEUE_URL"]
    sqs = boto3.client("sqs")

    processed = 0
    published = 0
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
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(asdict(notification)))
        published += 1

    return {"processed": processed, "published": published, "skipped": skipped}
