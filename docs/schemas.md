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

### `detections_3d` v1.0

One row per predicted 3D bounding box (lidar frames only).

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `detection_id` | `string` | yes | Unique detection identifier (UUID) |
| `frame_id` | `string` | yes | `frames.frame_id` this detection belongs to |
| `class_id` | `int32` | yes | Predicted class index |
| `class_name` | `string` | yes | Human-readable class label |
| `score` | `float32` | yes | Objectness confidence, `[0, 1]` |
| `center_xyz` | `fixed_size_list<double>[3]` | yes | Box center in meters, ego frame |
| `dimensions_whl` | `fixed_size_list<double>[3]` | yes | Box size `[width, height, length]` in meters |
| `yaw` | `float32` | yes | Heading angle (radians) around the vertical axis |
| `model_version` | `string` | yes | Checkpoint identifier, or `untrained-random-init` |

**Pydantic model:** `forge.schemas.detections_3d.Detection3DRecord`
**Table class:** `forge.schemas.detections_3d.Detections3DTable`
**Introduced:** Phase 3

### `tracks` v1.0

One row per detection, tagged with the track it was assigned to.

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `track_id` | `string` | yes | Globally unique — scoped as `{scene_id}:{sensor_id}:track-NNNNNN` |
| `detection_id` | `string` | yes | `detections_2d.detection_id` this row wraps |
| `frame_id` | `string` | yes | `frames.frame_id` this detection belongs to |
| `scene_id` | `string` | yes | Tracks never span scenes |
| `sensor_id` | `string` | yes | Sensor channel (e.g. `CAM_FRONT`) |
| `timestamp_us` | `int64` | yes | Frame timestamp (microseconds) |
| `class_id` | `int32` | yes | Detection's predicted class index |
| `class_name` | `string` | yes | Detection's predicted class label |
| `bbox_xyxy` | `fixed_size_list<double>[4]` | yes | `[x1, y1, x2, y2]` in pixel coordinates |
| `score` | `float32` | yes | Detection confidence score |
| `track_age` | `int32` | yes | Consecutive frames this track has matched (hit streak) |
| `tracker_version` | `string` | yes | Tracker run/config identifier, e.g. `sort-v1` |

**Pydantic model:** `forge.schemas.tracks.TrackRecord`
**Table class:** `forge.schemas.tracks.TracksTable`
**Introduced:** Phase 4

### `fused_objects` v1.0

One row per object after camera/lidar fusion — every camera and lidar
detection ends up in exactly one row, tagged by how it was resolved.

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `fusion_id` | `string` | yes | Unique identifier for this row (UUID) |
| `scene_id` | `string` | yes | Scene the fusion happened within |
| `timestamp_us` | `int64` | yes | Synchronized sample timestamp (microseconds) |
| `fusion_type` | `string` | yes | `matched`, `camera_only`, or `lidar_only` |
| `frame_id_2d` | `string` | yes | Camera frame_id, or `""` if no 2D side |
| `frame_id_3d` | `string` | yes | Lidar frame_id, or `""` if no 3D side |
| `detection_id_2d` | `string` | yes | `detections_2d.detection_id`, or `""` |
| `detection_id_3d` | `string` | yes | `detections_3d.detection_id`, or `""` |
| `class_id` | `int32` | yes | Class index, preferring the 3D side when both exist |
| `class_name` | `string` | yes | Human-readable class label |
| `score` | `float32` | yes | Confidence, preferring the 3D side |
| `bbox_xyxy` | `fixed_size_list<double>[4]` | yes | Camera box if present, else the projected lidar box, else `[0,0,0,0]` |
| `center_xyz` | `fixed_size_list<double>[3]` | yes | 3D center (ego frame); `[0,0,0]` if no 3D side |
| `dimensions_whl` | `fixed_size_list<double>[3]` | yes | 3D size; `[0,0,0]` if no 3D side |
| `yaw` | `float32` | yes | Heading in radians; `0.0` if no 3D side |
| `fuser_version` | `string` | yes | Fusion run/config identifier, e.g. `geometric-projection-v1` |

**Pydantic model:** `forge.schemas.fused_objects.FusedObjectRecord`
**Table class:** `forge.schemas.fused_objects.FusedObjectsTable`
**Introduced:** Phase 5

### `pseudo_labels` v1.0

One row per fused object, scored and routed by the active-learning /
pseudo-labeling policy.

| Column | PyArrow Type | Required | Description |
|--------|--------------|----------|-------------|
| `pseudo_label_id` | `string` | yes | Unique identifier for this row (UUID) |
| `fusion_id` | `string` | yes | Source `fused_objects.fusion_id` |
| `scene_id` | `string` | yes | Scene this object belongs to |
| `timestamp_us` | `int64` | yes | Sample timestamp (microseconds) |
| `fusion_type` | `string` | yes | Carried through: `matched`/`camera_only`/`lidar_only` |
| `class_id` | `int32` | yes | Class index |
| `class_name` | `string` | yes | Human-readable class label |
| `bbox_xyxy` | `fixed_size_list<double>[4]` | yes | Carried through from `fused_objects` |
| `center_xyz` | `fixed_size_list<double>[3]` | yes | Carried through from `fused_objects` |
| `dimensions_whl` | `fixed_size_list<double>[3]` | yes | Carried through from `fused_objects` |
| `yaw` | `float32` | yes | Carried through from `fused_objects` |
| `trust_score` | `float32` | yes | Cross-modal-agreement-adjusted confidence, `[0,1]` |
| `decision` | `string` | yes | `auto_accept`, `needs_review`, or `rejected` |
| `review_priority` | `float32` | yes | Binary entropy of `trust_score` — higher means review sooner |
| `labeler_version` | `string` | yes | Labeling run/config identifier, e.g. `trust-threshold-v1` |

**Pydantic model:** `forge.schemas.pseudo_labels.PseudoLabelRecord`
**Table class:** `forge.schemas.pseudo_labels.PseudoLabelsTable`
**Introduced:** Phase 6

## Future Tables (not yet built)

The following tables will be designed in their implementing phases — no stubs:

- `eval_metrics` — Phase 7

See [KNOWN_GAPS.md](../KNOWN_GAPS.md) for deferred work.
