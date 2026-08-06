# Glue catalog + Athena over the *processed* Parquet lake (the tables
# forge itself writes: frames, detections_2d, pseudo_labels, etc.) --
# distinct from main.tf's raw_data bucket, which holds the raw nuScenes
# uploads Lambda watches.
#
# Scope note: only `pseudo_labels` gets a full Glue table definition here
# (arguably the most useful one to query directly -- "what did the
# pipeline auto-label vs reject, and why"). Every other lake table would
# follow the identical mechanical pattern (map forge's Arrow schema to
# Glue/Hive column types, same Parquet SerDe) -- deferred as repetitive
# follow-up work, not built out for all ~10 tables in this pass. See
# KNOWN_GAPS.md.

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

# Mirrors forge.schemas.pseudo_labels.PseudoLabelsTable's Arrow schema,
# translated to Glue/Hive column types.
resource "aws_glue_catalog_table" "pseudo_labels" {
  name          = "pseudo_labels"
  database_name = aws_glue_catalog_database.forge_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.processed_lake.bucket}/pseudo_labels/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    columns {
      name = "pseudo_label_id"
      type = "string"
    }
    columns {
      name = "fusion_id"
      type = "string"
    }
    columns {
      name = "scene_id"
      type = "string"
    }
    columns {
      name = "timestamp_us"
      type = "bigint"
    }
    columns {
      name = "fusion_type"
      type = "string"
    }
    columns {
      name = "class_id"
      type = "int"
    }
    columns {
      name = "class_name"
      type = "string"
    }
    columns {
      name = "bbox_xyxy"
      type = "array<double>"
    }
    columns {
      name = "center_xyz"
      type = "array<double>"
    }
    columns {
      name = "dimensions_whl"
      type = "array<double>"
    }
    columns {
      name = "yaw"
      type = "float"
    }
    columns {
      name = "trust_score"
      type = "float"
    }
    columns {
      name = "decision"
      type = "string"
    }
    columns {
      name = "review_priority"
      type = "float"
    }
    columns {
      name = "labeler_version"
      type = "string"
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
