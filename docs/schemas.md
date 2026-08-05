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

### `frames` v1.1

One row per sensor sample timestamp in the data lake index.

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `frame_id` | `string` | yes | Unique frame identifier |
| `scene_id` | `string` | yes | Scene or log segment identifier |
| `timestamp_us` | `int64` | yes | Sample timestamp (microseconds) |
| `sensor_id` | `string` | yes | Sensor channel (e.g. `CAM_FRONT`) |
| `dataset_split` | `string` | yes | Split label: train / val / test / unknown |
| `data_path` | `string` | yes | Raw sensor file path, relative to the dataset root |
| `ingested_at` | `timestamp[us, tz=UTC]` | yes | UTC write timestamp |

**Pydantic model:** `forge.schemas.frames.FrameRecord`
**Table class:** `forge.schemas.frames.FramesTable`
**Introduced:** Phase 0

#### Migrations

- **v1.0 → v1.1** (Phase 1): added `data_path` (non-nullable, defaults to `""`
  for any pre-existing rows). Additive — no reader changes required.

### `calibration` v1.0

One row per unique calibrated sensor (deduplicated across frames).

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `token` | `string` | yes | Unique calibrated-sensor identifier |
| `sensor_id` | `string` | yes | Sensor channel |
| `translation` | `fixed_size_list<double>[3]` | yes | Sensor-to-ego translation `[x, y, z]` |
| `rotation` | `fixed_size_list<double>[4]` | yes | Sensor-to-ego quaternion `[w, x, y, z]` |
| `camera_intrinsic` | `list<double>` | yes | Flattened 3x3 intrinsic matrix; empty for non-cameras |

**Pydantic model:** `forge.schemas.calibration.CalibrationRecord`
**Table class:** `forge.schemas.calibration.CalibrationTable`
**Introduced:** Phase 1

### `ego_pose` v1.0

One row per unique vehicle pose (deduplicated across frames).

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `token` | `string` | yes | Unique ego-pose identifier |
| `timestamp_us` | `int64` | yes | Pose timestamp (microseconds) |
| `translation` | `fixed_size_list<double>[3]` | yes | Global-frame translation `[x, y, z]` |
| `rotation` | `fixed_size_list<double>[4]` | yes | Global-frame quaternion `[w, x, y, z]` |

**Pydantic model:** `forge.schemas.ego_pose.EgoPoseRecord`
**Table class:** `forge.schemas.ego_pose.EgoPoseTable`
**Introduced:** Phase 1

### `detections_2d` v1.0

One row per predicted 2D bounding box (camera frames only).

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `detection_id` | `string` | yes | Unique detection identifier (UUID) |
| `frame_id` | `string` | yes | `frames.frame_id` this detection belongs to |
| `class_id` | `int32` | yes | Predicted class index |
| `class_name` | `string` | yes | Human-readable class label |
| `score` | `float32` | yes | Model confidence, `[0, 1]` |
| `bbox_xyxy` | `fixed_size_list<double>[4]` | yes | `[x1, y1, x2, y2]` in pixel coordinates |
| `model_version` | `string` | yes | Checkpoint identifier, or `untrained-random-init` |

**Pydantic model:** `forge.schemas.detections_2d.Detection2DRecord`
**Table class:** `forge.schemas.detections_2d.Detections2DTable`
**Introduced:** Phase 2

## Future Tables (not yet built)

The following tables will be designed in their implementing phases — no stubs:

- `detections_3d` — Phase 3
- `tracks` — Phase 4
- `fused_objects` — Phase 5
- `pseudo_labels` — Phase 6
- `eval_metrics` — Phase 7

See [KNOWN_GAPS.md](../KNOWN_GAPS.md) for deferred work.
