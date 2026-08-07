# Phase 9 Completion — Infrastructure (partial: Ray, Lambda, Glue/Athena, EventBridge, Step Functions, ECS)

## Scope

Phase 9 is infrastructure, not a `forge` CLI command: Ray distributed
execution mode, and Terraform-provisioned AWS (S3 lake + Glue/Athena
catalog + Lambda + EventBridge + Step Functions + ECS, added across three
rounds — part of the same cloud-infrastructure requirement bucket as
S3/Athena in `ARCHITECTURE.md`'s requirement coverage map). **This is
still a partial phase completion** — labeled 🟡 in `README.md`'s phase
table, not ✅. What's built and what's still open are both listed
explicitly below.

## What was built

### Ray (local distributed execution)

- **`forge.distributed.run_distributed_map`** — wraps `ray.init`/
  `ray.remote`/`ray.get`/`ray.shutdown` for a local (no-cluster,
  no-cloud-spend) parallel map over CPU cores. Falls back to a plain
  sequential loop when `distributed=False` — identical results either
  way, just execution strategy.
- **Wired into `detect2d`'s inference path** — `_infer_one_frame` is now
  a standalone per-frame function (pure enough to run in a Ray worker),
  and `forge detect2d --mode infer --distributed` runs it across local
  CPU cores instead of `--local`'s sequential loop.
- **Honest verification caveat (important):** real multi-process Ray
  could not be run in the development sandbox that built this — three
  different configurations all hang/crash in Ray's C++ core. The
  sequential path was verified for real end-to-end via the actual CLI.
  The `distributed=True` path is verified by **mocking Ray's API
  boundary** (`tests/test_distributed.py`, 7 tests) — confirming this
  module's own logic is correct, not that Ray's real parallel execution
  works in this specific environment. See DECISIONS.md ADR-026 for the
  full investigation. **`--distributed` needs verification on a machine
  without this sandbox's constraint before being trusted in production.**

### Lambda (from the prior commit in this phase)

- `infra/lambda/ingest_trigger/handler.py`: S3-upload-triggered,
  validates the nuScenes-devkit layout, publishes to SQS.

### Terraform: Glue/Athena catalog

- `infra/terraform/glue_athena.tf`: a processed-lake S3 bucket, a Glue
  catalog database, one full Glue table definition (`pseudo_labels`,
  mapped from its real Arrow schema), and an Athena workgroup + results
  bucket.
- Deliberately scoped to one representative table rather than all ~10
  lake tables — the rest is the identical mechanical pattern, documented
  as follow-up in KNOWN_GAPS.md, not built out this round.

### EventBridge + Step Functions + ECS (this round)

- **`infra/terraform/ecs.tf`** — one ECS Fargate cluster, one shared
  `forge-pipeline` task definition (execution role, task role scoped to
  read the raw bucket + read/write the processed lake, CloudWatch
  logging). No per-stage task definitions — every pipeline stage runs
  the same `forge` CLI image, differentiated only by the command Step
  Functions overrides at `RunTask` time. See DECISIONS.md ADR-035.
- **`infra/terraform/step_functions.tf`** — a state machine chaining all
  eight pipeline stages (ingest → detect2d → detect3d → fuse → label →
  evaluate → curate → visualize) as `ecs:runTask.sync` Task states, each
  catching failures into a `PipelineFailed` state. Generated from a single
  `local.forge_pipeline_stages` list via HCL's `for`/`merge()` — verified
  structurally correct (every `Next`/`Catch`/`StartAt` reference
  resolves) by the new `scripts/validate_state_machine.py`, since no
  AWS-official ASL validator package exists on PyPI (checked before
  writing this).
- **`infra/terraform/eventbridge.tf`** — a custom event bus
  (`forge-events-<env>`), a rule matching the Lambda's exact published
  event shape, and a target starting the state machine.
- **`infra/lambda/ingest_trigger/handler.py`** — now publishes to *both*
  SQS (unchanged) and EventBridge (new) on every valid upload, with
  identical payloads. This closes the "SQS queue has no consumer" gap
  from the prior round — EventBridge/Step Functions is the real consumer
  now, chosen over the originally-planned custom polling worker because
  Step Functions' native ECS integration makes a bespoke poller
  redundant. See DECISIONS.md ADR-034.
- **New test coverage**: `tests/test_lambda_ingest_trigger.py` now mocks
  both the SQS and EventBridge boto3 clients (distinguished by service
  name), with dedicated tests for the EventBridge entry's shape and that
  both destinations receive identical payloads.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge                              # strict, 0 errors
uv run mypy infra/lambda/ingest_trigger/handler.py  # strict, 0 errors
uv run pytest -q                                     # see commit for exact numbers
```

Plus: every `.tf` file (9 total, including the three new ones) parsed
successfully with `python-hcl2`. The Step Functions ASL definition's
`locals`-block logic was independently re-simulated in Python (not just
trusted from HCL syntax validity) to confirm the generated state chain has
no dangling references before `scripts/validate_state_machine.py` was
written as a permanent, reusable version of that same check. All 20
Lambda tests (SQS + EventBridge) pass with a mocked boto3 client
distinguishing the two service names.

## Post-completion fixes and extensions (found via real user testing)

- **Rich markup bug**: every "install the [X] extra" CLI error message
  silently dropped the extra name (Rich interprets `[X]` as unrecognized
  style markup). Fixed by switching to `'X'` quotes across all six
  occurrences, verified directly against Rich's own console output.
- **Ray re-serialization**: real multi-process Ray confirmed working on a
  user's machine — the first real (non-mocked) confirmation — and
  surfaced a genuine inefficiency: `detect2d`'s model was captured in a
  Ray remote-function closure (72 MiB re-serialized per call) instead of
  going through `ray.put()`. Fixed by adding `shared_args` to
  `run_distributed_map`. The user's own follow-up run confirmed the 72 MiB
  warning is gone.
- **Test-assertion whitespace bug**: `test_track_requires_frames_lake` and
  `test_fuse_requires_frames_lake` failed on the user's machine because
  Rich word-wrapped a long line (containing a `tmp_path`), inserting a
  literal newline mid-phrase (`"forge\ningest"`) — a different mechanism
  from the earlier ANSI-fragmentation bug, needing a different fix.
  `tests/test_cli.py`'s `_plain()` helper now also collapses whitespace
  runs, not just strips ANSI codes.
- **Ray extended to `detect3d`**: `forge detect3d` now also accepts
  `--distributed`, using the identical (already-fixed) `shared_args`
  pattern — proving `run_distributed_map` is genuinely reusable, not
  detect2d-specific.

## What this phase does *not* claim

- Ray is wired into `detect2d`, `detect3d`, `track`, `fuse`, `label`, and
  `evaluate` — `curate` deliberately does not have a `--distributed` mode
  (its incremental LanceDB dedup has a real sequential data dependency
  between iterations, see DECISIONS.md ADR-037).
- Ray's actual parallel execution was never verified working in *this*
  development environment — only its API usage, via mocks. (It *has* been
  confirmed working on a real user machine, including the `ray.put()` fix
  above — but not here.)
- The Glue catalog covers `pseudo_labels` only, not the other ~9 lake
  tables.
- Step Functions starts a pipeline run on *every* validated upload, not
  once a whole dataset has finished landing — no completeness-tracking
  design was built this round (a real design decision, not an oversight
  — see DECISIONS.md ADR-034).
- Step Functions' Task states have no retry policy for transient
  ECS/Fargate failures — only genuine stage failures are caught (routed
  to `PipelineFailed`).
- `ecs.tf`'s task definition references a container image
  (`var.forge_container_image`) that was never built or pushed to a
  registry — a real deployment step outside this repo's cost-safety
  policy.
- None of the Lambda, EventBridge, Step Functions, or ECS infrastructure
  has ever been deployed or invoked against real AWS (`terraform apply`
  is never run here, by policy) — only unit-tested with mocked boto3
  (Lambda) or structurally validated (Terraform/ASL).
- Terraform files are HCL-syntax-valid (`python-hcl2`) and the ASL
  definition is structurally valid (`scripts/validate_state_machine.py`),
  not checked against the real AWS provider schema or the official
  `aws stepfunctions validate-state-machine-definition` API — no
  `terraform` binary or AWS-official ASL validator was available in the
  environment that built this.

## Known gaps carried forward

See `KNOWN_GAPS.md` for the full list and `ARCHITECTURE.md` for how this
phase maps back to the platform's requirement-coverage table.
