# S3-upload-triggered Lambda: validates nuScenes file layout, publishes
# valid uploads to SQS for a downstream Ray/ECS ingest worker to consume.
# See infra/lambda/ingest_trigger/handler.py for the actual logic.
#
# Lambda deliberately does none of the heavy lifting here -- it has
# execution time/memory limits unsuitable for the ML pipeline itself.
# This is the "notify something happened" layer; Ray (the other half of
# Phase 9) is the "do the actual work" layer.

resource "aws_sqs_queue" "ingest_notifications" {
  name                      = "forge-ingest-notifications-${var.environment}"
  message_retention_seconds = 4 * 24 * 60 * 60 # 4 days
  visibility_timeout_seconds = 300

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

data "archive_file" "ingest_trigger_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda/ingest_trigger/handler.py"
  output_path = "${path.module}/.build/ingest_trigger.zip"
}

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest_trigger" {
  name               = "forge-ingest-trigger-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "ingest_trigger_permissions" {
  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.aws_region}:*:log-group:/aws/lambda/forge-ingest-trigger-${var.environment}*"]
  }

  statement {
    sid       = "PublishIngestNotifications"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.ingest_notifications.arn]
  }
}

resource "aws_iam_role_policy" "ingest_trigger" {
  name   = "forge-ingest-trigger-${var.environment}"
  role   = aws_iam_role.ingest_trigger.id
  policy = data.aws_iam_policy_document.ingest_trigger_permissions.json
}

resource "aws_lambda_function" "ingest_trigger" {
  function_name = "forge-ingest-trigger-${var.environment}"
  role          = aws_iam_role.ingest_trigger.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 128

  filename         = data.archive_file.ingest_trigger_zip.output_path
  source_code_hash = data.archive_file.ingest_trigger_zip.output_base64sha256

  environment {
    variables = {
      INGEST_QUEUE_URL = aws_sqs_queue.ingest_notifications.url
    }
  }

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "ingest_trigger" {
  name              = "/aws/lambda/${aws_lambda_function.ingest_trigger.function_name}"
  retention_in_days = var.lambda_log_retention_days
}

resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest_trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.raw_data.arn
}

resource "aws_s3_bucket_notification" "raw_data_upload" {
  bucket = aws_s3_bucket.raw_data.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest_trigger.arn
    events               = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_lambda_permission.allow_s3]
}
