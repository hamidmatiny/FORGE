# Glue catalog + Athena over the *processed* Parquet lake (the tables
# forge itself writes: frames, detections_2d, pseudo_labels, etc.) --
# distinct from main.tf's raw_data bucket, which holds the raw nuScenes
# uploads Lambda watches.
#
# Every one of forge's 11 lake tables gets a real Glue table definition
# here, generated via for_each from local.lake_tables rather than 11
# hand-written near-duplicate resource blocks -- each column list below
# is copied directly from that table's real PyArrow arrow_schema() in
# src/forge/schemas/, not approximated. See DECISIONS.md.

resource "aws_s3_bucket" "processed_lake" {
  bucket = "${var.raw_data_bucket_name}-processed-lake"

  tags = {
    Project     = "forge"
    Environment = var.environment
    Purpose     = "processed-parquet-lake"
  }
}

resource "aws_s3_bucket_public_access_block" "processed_lake" {
  bucket = aws_s3_bucket.processed_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_glue_catalog_database" "forge_lake" {
  name = "forge_lake_${var.environment}"
}

locals {
  # One entry per lake table -- column (name, Glue/Hive type) pairs
  # mirror each table's real PyArrow arrow_schema() exactly:
  #   pa.string()                     -> "string"
  #   pa.int32()                      -> "int"
  #   pa.int64()                      -> "bigint"
  #   pa.float32()                    -> "float"
  #   pa.bool_()                      -> "boolean"
  #   pa.list_(pa.float64(), N or -)  -> "array<double>" (Hive doesn't
  #                                        distinguish fixed vs. variable
  #                                        length lists)
  #   pa.timestamp("us", tz="UTC")    -> "timestamp"
  lake_tables = {
    frames = [
      { name = "frame_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "sensor_id", type = "string" },
      { name = "dataset_split", type = "string" },
      { name = "data_path", type = "string" },
      { name = "ingested_at", type = "timestamp" },
    ]
    calibration = [
      { name = "token", type = "string" },
      { name = "sensor_id", type = "string" },
      { name = "translation", type = "array<double>" },
      { name = "rotation", type = "array<double>" },
      { name = "camera_intrinsic", type = "array<double>" },
    ]
    ego_pose = [
      { name = "token", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "translation", type = "array<double>" },
      { name = "rotation", type = "array<double>" },
    ]
    detections_2d = [
      { name = "detection_id", type = "string" },
      { name = "frame_id", type = "string" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "score", type = "float" },
      { name = "bbox_xyxy", type = "array<double>" },
      { name = "model_version", type = "string" },
    ]
    detections_3d = [
      { name = "detection_id", type = "string" },
      { name = "frame_id", type = "string" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "score", type = "float" },
      { name = "center_xyz", type = "array<double>" },
      { name = "dimensions_whl", type = "array<double>" },
      { name = "yaw", type = "float" },
      { name = "model_version", type = "string" },
    ]
    tracks = [
      { name = "track_id", type = "string" },
      { name = "detection_id", type = "string" },
      { name = "frame_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "sensor_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "bbox_xyxy", type = "array<double>" },
      { name = "score", type = "float" },
      { name = "track_age", type = "int" },
      { name = "tracker_version", type = "string" },
    ]
    fused_objects = [
      { name = "fusion_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "fusion_type", type = "string" },
      { name = "frame_id_2d", type = "string" },
      { name = "frame_id_3d", type = "string" },
      { name = "detection_id_2d", type = "string" },
      { name = "detection_id_3d", type = "string" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "score", type = "float" },
      { name = "bbox_xyxy", type = "array<double>" },
      { name = "center_xyz", type = "array<double>" },
      { name = "dimensions_whl", type = "array<double>" },
      { name = "yaw", type = "float" },
      { name = "fuser_version", type = "string" },
    ]
    pseudo_labels = [
      { name = "pseudo_label_id", type = "string" },
      { name = "fusion_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "fusion_type", type = "string" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "bbox_xyxy", type = "array<double>" },
      { name = "center_xyz", type = "array<double>" },
      { name = "dimensions_whl", type = "array<double>" },
      { name = "yaw", type = "float" },
      { name = "trust_score", type = "float" },
      { name = "decision", type = "string" },
      { name = "review_priority", type = "float" },
      { name = "labeler_version", type = "string" },
    ]
    ground_truth = [
      { name = "annotation_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "category_name", type = "string" },
      { name = "center_xyz", type = "array<double>" },
      { name = "dimensions_whl", type = "array<double>" },
      { name = "yaw", type = "float" },
      { name = "num_lidar_pts", type = "int" },
    ]
    eval_metrics = [
      { name = "eval_run_id", type = "string" },
      { name = "class_name", type = "string" },
      { name = "num_gt", type = "int" },
      { name = "num_predictions", type = "int" },
      { name = "num_matched", type = "int" },
      { name = "precision", type = "float" },
      { name = "recall", type = "float" },
      { name = "f1", type = "float" },
      { name = "average_precision", type = "float" },
      { name = "distance_threshold_m", type = "float" },
      { name = "eval_version", type = "string" },
    ]
    curated = [
      { name = "pseudo_label_id", type = "string" },
      { name = "scene_id", type = "string" },
      { name = "timestamp_us", type = "bigint" },
      { name = "class_id", type = "int" },
      { name = "class_name", type = "string" },
      { name = "bbox_xyxy", type = "array<double>" },
      { name = "center_xyz", type = "array<double>" },
      { name = "dimensions_whl", type = "array<double>" },
      { name = "yaw", type = "float" },
      { name = "trust_score", type = "float" },
      { name = "is_duplicate", type = "boolean" },
      { name = "duplicate_of_id", type = "string" },
      { name = "curation_version", type = "string" },
    ]
  }
}

resource "aws_glue_catalog_table" "lake" {
  for_each = local.lake_tables

  name          = each.key
  database_name = aws_glue_catalog_database.forge_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.processed_lake.bucket}/${each.key}/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    dynamic "columns" {
      for_each = each.value
      content {
        name = columns.value.name
        type = columns.value.type
      }
    }
  }
}

resource "aws_s3_bucket" "athena_results" {
  bucket = "${var.raw_data_bucket_name}-athena-results"

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

resource "aws_athena_workgroup" "forge" {
  name = "forge-${var.environment}"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results.bucket}/query-results/"
    }
  }

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}
