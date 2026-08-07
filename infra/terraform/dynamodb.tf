# Tracks which file categories (metadata_table / sensor_file) have been
# seen for each dataset_root, so the ingest-trigger Lambda can trigger the
# pipeline once per dataset when it first looks minimally complete,
# instead of on every single S3 upload. See
# infra/lambda/ingest_trigger/handler.py's _check_and_record_completeness
# for the heuristic and its real limitations (documented there and in
# KNOWN_GAPS.md), and DECISIONS.md for why this design was chosen over a
# fully rigorous per-file manifest.

resource "aws_dynamodb_table" "dataset_completeness" {
  name         = "forge-dataset-completeness-${var.environment}"
  billing_mode = "PAY_PER_REQUEST" # no capacity to provision/pay for when never applied
  hash_key     = "dataset_root"

  attribute {
    name = "dataset_root"
    type = "S"
  }

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}
