# EventBridge: the link between the Lambda's validation ("this upload
# looks real") and Step Functions' orchestration ("run the pipeline").
# A custom event bus (not the account default) keeps forge's events
# scoped and easy to reason about independently of anything else running
# in the same AWS account.

resource "aws_cloudwatch_event_bus" "forge" {
  name = "forge-events-${var.environment}"

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_event_rule" "trigger_pipeline_on_valid_upload" {
  name           = "forge-trigger-pipeline-${var.environment}"
  event_bus_name = aws_cloudwatch_event_bus.forge.name

  # Matches the exact Source/DetailType the Lambda publishes -- see
  # infra/lambda/ingest_trigger/handler.py's _EVENT_SOURCE/_EVENT_DETAIL_TYPE.
  event_pattern = jsonencode({
    source      = ["forge.ingest"]
    detail-type = ["IngestUploadValidated"]
  })

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}

data "aws_iam_policy_document" "eventbridge_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eventbridge_start_pipeline" {
  name               = "forge-eventbridge-start-pipeline-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.eventbridge_assume_role.json
}

data "aws_iam_policy_document" "eventbridge_start_pipeline_permissions" {
  statement {
    sid       = "StartPipelineExecution"
    actions   = ["states:StartExecution"]
    resources = [aws_sfn_state_machine.forge_pipeline.arn]
  }
}

resource "aws_iam_role_policy" "eventbridge_start_pipeline" {
  name   = "forge-eventbridge-start-pipeline-${var.environment}"
  role   = aws_iam_role.eventbridge_start_pipeline.id
  policy = data.aws_iam_policy_document.eventbridge_start_pipeline_permissions.json
}

resource "aws_cloudwatch_event_target" "start_pipeline" {
  rule           = aws_cloudwatch_event_rule.trigger_pipeline_on_valid_upload.name
  event_bus_name = aws_cloudwatch_event_bus.forge.name
  arn            = aws_sfn_state_machine.forge_pipeline.arn
  role_arn       = aws_iam_role.eventbridge_start_pipeline.arn

  # The event's "detail" (the IngestNotification payload) becomes the
  # state machine's execution input, so the first Task state's
  # `Environment[0]."Value.$" = "$.dataset_root"` (step_functions.tf)
  # resolves correctly.
  input_path = "$.detail"
}
