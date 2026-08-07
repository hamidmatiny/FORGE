#!/usr/bin/env python3
"""Structural validation for infra/terraform/step_functions.tf's ASL definition.

No AWS-official Amazon States Language validator package was available in
the environment that built this (checked pip for asl-validator and
similar — none exist; see KNOWN_GAPS.md). This is NOT a substitute for
`aws stepfunctions validate-state-machine-definition` against a real
definition, which this was never run against — it only catches the class
of bugs that matter most for a hand-written chained state machine: dangling
`Next`/`Catch`/`StartAt` references, and terminal-state/Task-state shape
errors.

The stage list below must be kept in sync with step_functions.tf's
`local.forge_pipeline_stages` by hand — there's no shared source of
truth between HCL and Python here, since no `terraform` binary was
available to actually evaluate the `.tf` file's `locals` block (see
KNOWN_GAPS.md).
"""

from __future__ import annotations

import sys

FORGE_PIPELINE_STAGES = [
    "Ingest",
    "Detect2D",
    "Detect3D",
    "Fuse",
    "Label",
    "Evaluate",
    "Curate",
    "Visualize",
]


def build_definition() -> dict:
    """Mirror step_functions.tf's `local.forge_pipeline_definition` logic in Python."""
    retry = [
        {
            "ErrorEquals": ["States.Timeout", "States.TaskFailed", "ECS.AmazonECSException"],
            "IntervalSeconds": 30,
            "MaxAttempts": 2,
            "BackoffRate": 2.0,
        }
    ]
    states: dict[str, dict] = {}
    for idx, name in enumerate(FORGE_PIPELINE_STAGES):
        next_state = (
            FORGE_PIPELINE_STAGES[idx + 1]
            if idx < len(FORGE_PIPELINE_STAGES) - 1
            else "PipelineSucceeded"
        )
        states[name] = {
            "Type": "Task",
            "Resource": "arn:aws:states:::ecs:runTask.sync",
            "Next": next_state,
            "Retry": retry,
            "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "PipelineFailed"}],
        }
    states["PipelineSucceeded"] = {"Type": "Succeed"}
    states["PipelineFailed"] = {"Type": "Fail", "Error": "PipelineStageFailed"}
    return {"StartAt": FORGE_PIPELINE_STAGES[0], "States": states}


def validate(definition: dict) -> list[str]:
    """Return a list of structural errors; empty means valid."""
    errors: list[str] = []
    states = definition["States"]
    all_names = set(states.keys())

    if definition["StartAt"] not in all_names:
        errors.append(f"StartAt '{definition['StartAt']}' is not a defined state")

    for name, state in states.items():
        state_type = state.get("Type")

        if state_type in ("Succeed", "Fail"):
            if "Next" in state:
                errors.append(f"Terminal state '{name}' ({state_type}) must not have Next")
            continue

        if "Next" not in state and not state.get("End"):
            errors.append(f"Non-terminal state '{name}' has neither Next nor End")

        if "Next" in state and state["Next"] not in all_names:
            errors.append(f"State '{name}' Next -> '{state['Next']}' does not exist")

        for catch in state.get("Catch", []):
            if catch["Next"] not in all_names:
                errors.append(f"State '{name}' Catch -> '{catch['Next']}' does not exist")

        if state_type == "Task" and not state.get("Retry"):
            errors.append(f"Task state '{name}' has no Retry policy for transient failures")

    return errors


def main() -> int:
    definition = build_definition()
    errors = validate(definition)
    if errors:
        print(f"INVALID: {len(errors)} structural error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(FORGE_PIPELINE_STAGES)} pipeline stages, all references resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
