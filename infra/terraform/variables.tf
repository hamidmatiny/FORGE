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

variable "forge_container_image" {
  description = <<-EOT
    Container image URI for the forge CLI, used by every pipeline stage's
    ECS task (see ecs.tf). Must already exist in a registry (e.g. ECR) --
    this project's Docker image (Phase 0's Dockerfile) is never built or
    pushed automatically; that's a real deployment step outside this
    repo's cost-safety policy of never touching real cloud resources.
  EOT
  type        = string
}

variable "ecs_task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)."
  type        = number
  default     = 1024
}

variable "ecs_task_memory_mb" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 2048
}

variable "private_subnet_ids" {
  description = "Subnet IDs for Fargate tasks to run in. Must already exist -- no VPC is provisioned here."
  type        = list(string)
}
