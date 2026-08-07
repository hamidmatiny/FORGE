# ECS Fargate: the actual compute layer Step Functions orchestrates.
#
# One shared task definition ("forge-pipeline") rather than one per
# pipeline stage -- every stage is the same `forge` CLI image, just a
# different subcommand. Step Functions' ECS RunTask integration
# overrides the container command per state (see step_functions.tf),
# so this task definition only needs to exist once.
#
# Never deployed by this repo -- `terraform apply` is never run here, by
# policy (see DECISIONS.md). This defines what *would* run, not
# something running.

resource "aws_ecs_cluster" "forge" {
  name = "forge-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: what ECS itself needs (pull the image, write logs).
resource "aws_iam_role" "forge_pipeline_execution" {
  name               = "forge-pipeline-execution-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
}

resource "aws_iam_role_policy_attachment" "forge_pipeline_execution" {
  role       = aws_iam_role.forge_pipeline_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task role: what forge's own code needs at runtime (read the raw bucket,
# read/write the processed lake).
data "aws_iam_policy_document" "forge_pipeline_task_permissions" {
  statement {
    sid       = "ReadRawData"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.raw_data.arn, "${aws_s3_bucket.raw_data.arn}/*"]
  }

  statement {
    sid = "ReadWriteProcessedLake"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [aws_s3_bucket.processed_lake.arn, "${aws_s3_bucket.processed_lake.arn}/*"]
  }
}

resource "aws_iam_role" "forge_pipeline_task" {
  name               = "forge-pipeline-task-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
}

resource "aws_iam_role_policy" "forge_pipeline_task" {
  name   = "forge-pipeline-task-${var.environment}"
  role   = aws_iam_role.forge_pipeline_task.id
  policy = data.aws_iam_policy_document.forge_pipeline_task_permissions.json
}

resource "aws_cloudwatch_log_group" "forge_pipeline" {
  name              = "/ecs/forge-pipeline-${var.environment}"
  retention_in_days = var.lambda_log_retention_days
}

resource "aws_ecs_task_definition" "forge_pipeline" {
  family                   = "forge-pipeline-${var.environment}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory_mb
  execution_role_arn       = aws_iam_role.forge_pipeline_execution.arn
  task_role_arn            = aws_iam_role.forge_pipeline_task.arn

  container_definitions = jsonencode([
    {
      name  = "forge"
      image = var.forge_container_image
      # No command here -- every Step Functions Task state supplies its
      # own via containerOverrides (e.g. ["forge", "ingest", "--input-dir", ...]).
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.forge_pipeline.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "forge"
        }
      }
    }
  ])

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}
