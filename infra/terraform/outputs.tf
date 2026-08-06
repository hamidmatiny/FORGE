output "raw_data_bucket_name" {
  description = "S3 bucket that raw nuScenes uploads should land in."
  value       = aws_s3_bucket.raw_data.bucket
}

output "ingest_notifications_queue_url" {
  description = "SQS queue URL a downstream Ray/ECS ingest worker should consume from."
  value       = aws_sqs_queue.ingest_notifications.url
}

output "ingest_trigger_lambda_arn" {
  description = "ARN of the S3-upload-validating Lambda."
  value       = aws_lambda_function.ingest_trigger.arn
}

output "processed_lake_bucket_name" {
  description = "S3 bucket for the processed Parquet lake tables (frames, detections, pseudo_labels, etc.)."
  value       = aws_s3_bucket.processed_lake.bucket
}

output "glue_database_name" {
  description = "Glue catalog database Athena queries against."
  value       = aws_glue_catalog_database.forge_lake.name
}

output "athena_workgroup_name" {
  description = "Athena workgroup for forge lake queries."
  value       = aws_athena_workgroup.forge.name
}
