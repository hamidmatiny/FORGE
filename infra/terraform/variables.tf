variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment tag (e.g. dev, staging)."
  type        = string
  default     = "dev"
}

variable "raw_data_bucket_name" {
  description = "S3 bucket that raw nuScenes-format uploads land in. Must be globally unique."
  type        = string
}

variable "lambda_log_retention_days" {
  description = "CloudWatch log retention for the ingest-trigger Lambda."
  type        = number
  default     = 14
}
