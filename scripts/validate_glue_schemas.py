#!/usr/bin/env python3
"""Verify infra/terraform/glue_athena.tf's Glue table columns match the
real PyArrow schemas in src/forge/schemas/ exactly.

Written after ADR-038 expanded the Glue catalog to all 11 lake tables:
hand-copying each column list into Terraform is a real, silent-failure
risk (a wrong Glue type wouldn't be caught until someone queried the
table for real) with nothing to keep the two in sync automatically. This
closes that gap by computing the expected Arrow-to-Glue/Hive type mapping
programmatically from the actual schema classes, not by re-deriving the
mapping rules by hand a second time, and comparing against what's
actually in the .tf file.

Needs python-hcl2 (see scripts/check.sh, which installs it automatically
if missing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import hcl2
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forge.schemas import (  # noqa: E402
    CalibrationTable,
    CuratedTable,
    Detections2DTable,
    Detections3DTable,
    EgoPoseTable,
    EvalMetricsTable,
    FramesTable,
    FusedObjectsTable,
    GroundTruthTable,
    PseudoLabelsTable,
    TracksTable,
)

TERRAFORM_PATH = Path(__file__).parent.parent / "infra" / "terraform" / "glue_athena.tf"

TABLES: dict[str, type] = {
    "frames": FramesTable,
    "calibration": CalibrationTable,
    "ego_pose": EgoPoseTable,
    "detections_2d": Detections2DTable,
    "detections_3d": Detections3DTable,
    "tracks": TracksTable,
    "fused_objects": FusedObjectsTable,
    "pseudo_labels": PseudoLabelsTable,
    "ground_truth": GroundTruthTable,
    "eval_metrics": EvalMetricsTable,
    "curated": CuratedTable,
}


def arrow_type_to_glue(arrow_type: pa.DataType) -> str:
    """Mirror glue_athena.tf's documented Arrow -> Glue/Hive type mapping."""
    if pa.types.is_string(arrow_type):
        return "string"
    if pa.types.is_int32(arrow_type):
        return "int"
    if pa.types.is_int64(arrow_type):
        return "bigint"
    if pa.types.is_float32(arrow_type):
        return "float"
    if pa.types.is_float64(arrow_type):
        return "double"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return f"array<{arrow_type_to_glue(arrow_type.value_type)}>"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    raise ValueError(f"No Glue type mapping defined for Arrow type: {arrow_type}")


def _strip_quotes(value: object) -> object:
    """python-hcl2 sometimes leaves string literals wrapped in literal quote characters."""
    return value.strip('"') if isinstance(value, str) else value


def main() -> int:
    expected: dict[str, list[tuple[str, str]]] = {
        name: [(f.name, arrow_type_to_glue(f.type)) for f in table_cls.arrow_schema()]
        for name, table_cls in TABLES.items()
    }

    with TERRAFORM_PATH.open() as f:
        parsed = hcl2.load(f)
    actual_tables = parsed["locals"][0]["lake_tables"]

    errors: list[str] = []

    missing = set(expected) - set(actual_tables)
    extra = set(actual_tables) - set(expected)
    if missing:
        errors.append(f"glue_athena.tf is missing table(s): {sorted(missing)}")
    if extra:
        errors.append(f"glue_athena.tf has table(s) not in forge.schemas: {sorted(extra)}")

    for name, expected_cols in expected.items():
        if name not in actual_tables:
            continue
        actual_cols = [
            (_strip_quotes(c["name"]), _strip_quotes(c["type"])) for c in actual_tables[name]
        ]
        if actual_cols != expected_cols:
            errors.append(
                f"{name}: column mismatch\n  expected: {expected_cols}\n  actual:   {actual_cols}"
            )

    if errors:
        print(f"INVALID: {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"OK: all {len(expected)} Glue table definitions match their real PyArrow schemas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
