# Phase 9 Completion — Infrastructure (partial: Lambda)

## Scope

Phase 9 is infrastructure, not a `forge` CLI command: Ray distributed
execution mode, and Terraform-provisioned AWS (S3 lake + Glue/Athena
catalog), per `ARCHITECTURE.md`. **This round adds Lambda** — part of
the same cloud-infrastructure requirement bucket as S3/Athena (see
`ARCHITECTURE.md`'s requirement coverage map), which the original Phase 9
plan had covered except Lambda specifically. **Ray and the Glue/Athena
catalog are still open** — this is a partial phase completion, not a
full one, and is labeled that way in `README.md`'s phase table (🟡, not
✅).

## What was built

- **`infra/lambda/ingest_trigger/handler.py`** — a real, tested Lambda:
  triggered on S3 `ObjectCreated` events for the raw-data bucket, it
  validates the uploaded key against the nuScenes-devkit layout (the same
  convention `forge.ingest.nuscenes` uses: `<version>/*.json` or
  `samples|sweeps/**`, optionally under an arbitrary prefix for
  multi-dataset buckets) and publishes a structured notification to SQS.
  Deliberately does none of the actual pipeline work — Lambda's execution
  time/memory limits make it the wrong tool for that; it's the
  "notify," not the "compute," layer (Ray/ECS are the compute layer, per
  ADR-024).
- **`infra/terraform/`** — `versions.tf`, `variables.tf`, `main.tf` (the
  S3 raw-data bucket, encrypted, public access blocked), `lambda.tf` (the
  Lambda function, its IAM role/policy, the SQS queue it publishes to,
  the S3 event-notification wiring), `outputs.tf`. Applied out-of-band
  only — never in CI, matching every sibling repo's cost-safety policy.
- **9 pytest test cases → 18 collected test runs** (several parametrized)
  in `tests/test_lambda_ingest_trigger.py`: key-validation for both valid
  and invalid layouts, notification-building (including the
  multi-dataset-prefix case), and the full handler with a mocked boto3
  SQS client — never touches real AWS.
- **`pyproject.toml`**: populated the Phase-0-era `aws = []` placeholder
  extra with `boto3`. CI gained a second `mypy` step
  (`infra/lambda/ingest_trigger/handler.py`) alongside the existing
  `mypy src/forge`, since this code lives outside the main package and
  isn't covered by that invocation.

## A real bug found and fixed during manual testing

The first version of the key-validation regex anchored the nuScenes
layout pattern to the start of the S3 key. Testing it against a
realistic multi-dataset bucket layout
(`fleet-a/2026-01/samples/CAM_FRONT/x.jpg`) immediately failed — a
genuinely valid nuScenes file, just not at the bucket root, got rejected.
Fixed by adding a non-greedy prefix-capture group to both regexes before
writing any tests on top of the broken version. Full writeup in
DECISIONS.md.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge                              # strict, 50 source files, 0 errors
uv run mypy infra/lambda/ingest_trigger/handler.py  # strict, 0 errors
uv run pytest -q                                     # 156 passed, 90.16% coverage (threshold 80%)
```

Plus: every `.tf` file parsed successfully with `python-hcl2` (syntax
validity only — no `terraform` binary was available in the environment
that built this, so full semantic validation against the AWS provider
schema wasn't possible; see KNOWN_GAPS.md). Manually exercised
`lambda_handler` against a synthetic S3 event with a mix of valid and
invalid keys and confirmed both the summary counts and the exact SQS
message body contents before writing the corresponding tests.

## What this phase does *not* claim

- Ray distributed execution mode and the Glue/Athena catalog (the rest of
  the original Phase 9 plan) are still open.
- Nothing consumes the SQS queue yet — that's the Ray/ECS ingest worker's
  job, which doesn't exist.
- The Lambda has never been deployed or invoked against real AWS
  (`terraform apply` is never run here, by policy) — only its logic is
  unit-tested with a mocked boto3 client.
- Terraform files are HCL-syntax-valid (`python-hcl2`), not
  `terraform validate`-checked against the real AWS provider schema.

## Known gaps carried forward

See `KNOWN_GAPS.md` for the full list and `ARCHITECTURE.md` for how this
phase maps back to the platform's requirement-coverage table.
