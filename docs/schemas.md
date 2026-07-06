# FORGE Schemas

Versioned Parquet schemas for the FORGE data lake. All pipeline stages read and write **only** through these schemas.

## Versioning Policy

1. Every table has a `SchemaVersion(major, minor)`.
2. **Minor** bump: additive, backward-compatible changes (new nullable columns).
3. **Major** bump: breaking changes (rename, type change, column removal). Requires a migration note below and updated readers/writers.
4. Parquet files embed metadata keys:
   - `forge.table` — logical table name (e.g. `frames`)
   - `forge.schema_version` — string form (e.g. `v1.0`)
5. No pickle as interchange format — ever.

## Tables

### `frames` v1.0

One row per sensor sample timestamp in the data lake index.

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `frame_id` | `string` | yes | Unique frame identifier |
| `scene_id` | `string` | yes | Scene or log segment identifier |
| `timestamp_us` | `int64` | yes | Sample timestamp (microseconds) |
| `sensor_id` | `string` | yes | Sensor channel (e.g. `CAM_FRONT`) |
| `dataset_split` | `string` | yes | Split label: train / val / test / unknown |
| `ingested_at` | `timestamp[us, tz=UTC]` | yes | UTC write timestamp |

**Pydantic model:** `forge.schemas.frames.FrameRecord`  
**Table class:** `forge.schemas.frames.FramesTable`  
**Introduced:** Phase 0

#### Migrations

_None yet._

## Future Tables (not in Phase 0)

The following tables will be designed in their implementing phases — no stubs:

- `detections_2d` — Phase 2
- `detections_3d` — Phase 3
- `tracks` — Phase 4
- `fused_objects` — Phase 5
- `eval_metrics` — Phase 6

See [KNOWN_GAPS.md](../KNOWN_GAPS.md) for deferred work.
