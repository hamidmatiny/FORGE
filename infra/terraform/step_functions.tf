# Step Functions: orchestrates the pipeline stages as a sequence of ECS
# RunTask.sync calls against the shared "forge-pipeline" task definition
# (ecs.tf), each overriding the container command to run one forge CLI
# subcommand. Triggered by the EventBridge rule in eventbridge.tf, which
# fires on every Lambda-validated upload (see ADR-034 for why "every
# upload triggers a run" was chosen over a completeness-tracking design).
#
# Never deployed by this repo -- `terraform apply` is never run here, by
# policy. The ASL definition below is validated for structural
# correctness (valid JSON, every Next/StartAt reference resolves to a
# real state) by scripts/validate_state_machine.py, since no AWS-official
# ASL validator package was available in the environment that built this
# (see KNOWN_GAPS.md) -- not a substitute for real `aws stepfunctions
# validate-state-machine-definition`, which this was never run against.

locals {
  # One Task state per pipeline stage. `command` is the forge CLI
  # invocation that stage runs. Every state gets the same Retry policy
  # (see forge_pipeline_retry below) for transient ECS/Fargate failures --
  # genuine application failures still exhaust retries and fall through
  # to Catch -> PipelineFailed.
  forge_pipeline_stages = [
    { name = "Ingest", command = ["forge", "ingest", "--input-dir", "$RAW_DATA_PATH", "--local"] },
    { name = "Detect2D", command = ["forge", "detect2d", "--mode", "infer", "--images-root", "$RAW_DATA_PATH", "--local"] },
    { name = "Detect3D", command = ["forge", "detect3d", "--mode", "infer", "--pointcloud-root", "$RAW_DATA_PATH", "--local"] },
    { name = "Fuse", command = ["forge", "fuse", "--local"] },
    { name = "Label", command = ["forge", "label", "--local"] },
    { name = "Evaluate", command = ["forge", "evaluate", "--gt-input-dir", "$RAW_DATA_PATH", "--local"] },
    { name = "Curate", command = ["forge", "curate", "--local"] },
    { name = "Visualize", command = ["forge", "visualize", "--format", "mcap", "--local"] },
  ]

  # Retries transient infra failures (ECS API throttling, a task that
  # times out waiting for capacity) up to 2 extra attempts with
  # exponential backoff -- deliberately NOT "States.ALL", which would
  # also retry genuine application failures (a real bug in that stage)
  # as if they might succeed on a second try. Numbers (30s initial
  # interval, 2.0 backoff rate, 2 retries) are a reasonable starting
  # point, not tuned against any real failure data -- there isn't any,
  # since this has never been deployed (see KNOWN_GAPS.md).
  forge_pipeline_retry = [
    {
      ErrorEquals     = ["States.Timeout", "States.TaskFailed", "ECS.AmazonECSException"]
      IntervalSeconds = 30
      MaxAttempts     = 2
      BackoffRate     = 2.0
    }
  ]

  # Build states with correct Next/End chaining from the list above.
  forge_pipeline_states = {
    for idx, stage in local.forge_pipeline_stages :
    stage.name => merge(
      {
        Type       = "Task"
        Resource   = "arn:aws:states:::ecs:runTask.sync"
        ResultPath = "$.${lower(stage.name)}Result"
        Retry      = local.forge_pipeline_retry
        Parameters = {
          Cluster        = aws_ecs_cluster.forge.arn
          TaskDefinition = aws_ecs_task_definition.forge_pipeline.arn
          LaunchType     = "FARGATE"
          NetworkConfiguration = {
            AwsvpcConfiguration = {
              Subnets        = var.private_subnet_ids
              AssignPublicIp = "DISABLED"
            }
          }
          Overrides = {
            ContainerOverrides = [
              {
                Name    = "forge"
                Command = stage.command
                Environment = [
                  { Name = "FORGE_DATA_LAKE_ROOT", "Value.$" = "$.dataset_root" }
                ]
              }
            ]
          }
        }
        Catch = [
          { ErrorEquals = ["States.ALL"], Next = "PipelineFailed" }
        ]
      },
      idx < length(local.forge_pipeline_stages) - 1
      ? { Next = local.forge_pipeline_stages[idx + 1].name }
      : { Next = "PipelineSucceeded" }
    )
  }

  forge_pipeline_definition = {
    Comment = "FORGE offline-perception pipeline: ingest through visualize, one ECS task per stage."
    StartAt = local.forge_pipeline_stages[0].name
    States = merge(
      local.forge_pipeline_states,
      {
        PipelineSucceeded = { Type = "Succeed" }
        PipelineFailed    = { Type = "Fail", Error = "PipelineStageFailed" }
      }
    )
  }
}

data "aws_iam_policy_document" "step_functions_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "forge_pipeline_state_machine" {
  name               = "forge-pipeline-state-machine-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.step_functions_assume_role.json
}

data "aws_iam_policy_document" "forge_pipeline_state_machine_permissions" {
  statement {
    sid       = "RunEcsTasks"
    actions   = ["ecs:RunTask", "ecs:StopTask", "ecs:DescribeTasks"]
    resources = [aws_ecs_task_definition.forge_pipeline.arn]
  }

  statement {
    sid       = "PassEcsRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.forge_pipeline_execution.arn, aws_iam_role.forge_pipeline_task.arn]
  }

  statement {
    sid = "TrackEcsEventsForSyncIntegration"
    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule",
    ]
    # The .sync ECS integration relies on an AWS-managed EventBridge rule
    # for task-completion callbacks; this is the standard IAM statement
    # AWS documents for that integration, scoped to the rule it manages.
    resources = ["arn:aws:events:${var.aws_region}:*:rule/StepFunctionsGetEventsForECSTaskRule"]
  }
}

resource "aws_iam_role_policy" "forge_pipeline_state_machine" {
  name   = "forge-pipeline-state-machine-${var.environment}"
  role   = aws_iam_role.forge_pipeline_state_machine.id
  policy = data.aws_iam_policy_document.forge_pipeline_state_machine_permissions.json
}

resource "aws_sfn_state_machine" "forge_pipeline" {
  name     = "forge-pipeline-${var.environment}"
  role_arn = aws_iam_role.forge_pipeline_state_machine.arn
  definition = jsonencode(local.forge_pipeline_definition)

  tags = {
    Project     = "forge"
    Environment = var.environment
  }
}
