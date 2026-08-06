"""Curate pseudo-labels: flag near-duplicates via LanceDB vector search."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from forge.curate.features import VECTOR_DIM, build_feature_vector
from forge.schemas import CuratedRecord, PseudoLabelRecord

CURATION_VERSION = "lancedb-geometric-dedup-v1"

_LANCE_SCHEMA = pa.schema(
    [
        pa.field("pseudo_label_id", pa.string()),
        pa.field("scene_id", pa.string()),
        pa.field("class_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ]
)


def _escape(value: str) -> str:
    """Escape single quotes for a LanceDB SQL-style filter string."""
    return value.replace("'", "''")


def run_curation(
    pseudo_labels: list[PseudoLabelRecord],
    lancedb_path: Path,
    distance_threshold: float = 1.0,
    decision_filter: str = "auto_accept",
) -> list[CuratedRecord]:
    """Flag near-duplicate pseudo-labels using an incremental LanceDB search.

    Candidates are processed highest-`trust_score`-first: the first of a
    group of near-duplicates encountered is kept, and every subsequent one
    within `distance_threshold` (in the 8-dim geometric feature space, see
    `forge.curate.features`) of an already-kept object *in the same scene
    and class* is flagged as a duplicate rather than dropped — the
    decision stays auditable, nothing silently disappears.

    Args:
        pseudo_labels: Rows from `forge label`.
        lancedb_path: Directory for the LanceDB database files.
        distance_threshold: Max feature-space (squared L2) distance to
            count as a near-duplicate.
        decision_filter: Which `pseudo_labels.decision` to curate (default
            `auto_accept` — curation builds the dataset you'd actually
            export). Pass `"all"` to curate every decision.
    """
    candidates = [
        p for p in pseudo_labels if decision_filter == "all" or p.decision == decision_filter
    ]
    candidates.sort(key=lambda p: p.trust_score, reverse=True)

    db = lancedb.connect(str(lancedb_path))
    table = db.create_table("curated_objects", schema=_LANCE_SCHEMA, mode="overwrite")

    output: list[CuratedRecord] = []

    for candidate in candidates:
        vector = build_feature_vector(candidate.center_xyz, candidate.dimensions_whl, candidate.yaw)
        duplicate_of_id = ""

        if table.count_rows() > 0:
            scene_filter = _escape(candidate.scene_id)
            class_filter = _escape(candidate.class_name)
            matches = (
                table.search(vector)
                .where(f"scene_id = '{scene_filter}' AND class_name = '{class_filter}'")
                .limit(1)
                .to_list()
            )
            if matches and matches[0]["_distance"] <= distance_threshold:
                duplicate_of_id = str(matches[0]["pseudo_label_id"])

        is_duplicate = duplicate_of_id != ""
        if not is_duplicate:
            table.add(
                [
                    {
                        "pseudo_label_id": candidate.pseudo_label_id,
                        "scene_id": candidate.scene_id,
                        "class_name": candidate.class_name,
                        "vector": vector,
                    }
                ]
            )

        output.append(
            CuratedRecord(
                pseudo_label_id=candidate.pseudo_label_id,
                scene_id=candidate.scene_id,
                timestamp_us=candidate.timestamp_us,
                class_id=candidate.class_id,
                class_name=candidate.class_name,
                bbox_xyxy=candidate.bbox_xyxy,
                center_xyz=candidate.center_xyz,
                dimensions_whl=candidate.dimensions_whl,
                yaw=candidate.yaw,
                trust_score=candidate.trust_score,
                is_duplicate=is_duplicate,
                duplicate_of_id=duplicate_of_id,
                curation_version=CURATION_VERSION,
            )
        )

    return output
