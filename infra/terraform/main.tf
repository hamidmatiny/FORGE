# Raw-data lake bucket. Uploads matching the nuScenes-devkit layout here
# trigger infra/lambda/ingest_trigger/handler.py (see lambda.tf).
#
# Scope note: this is the S3 half of Phase 9's "S3 lake + Glue/Athena
# catalog" plan (see ARCHITECTURE.md). The Glue catalog + Athena query
# setup for the *processed* Parquet lake tables is still open — tracked in
# KNOWN_GAPS.md, not silently assumed done by this bucket existing.

resource "aws_s3_bucket" "raw_data" {
  bucket = var.raw_data_bucket_name

  tags = {
    Project     = "forge"
    Environment = var.environment
    Purpose     = "raw-nuscenes-uploads"
  }
}

resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
