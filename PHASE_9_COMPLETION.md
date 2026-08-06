# Phase 9 Completion — Infrastructure (partial: Ray, Lambda, Glue/Athena)

## Scope

Phase 9 is infrastructure, not a `forge` CLI command: Ray distributed
execution mode, and Terraform-provisioned AWS (S3 lake + Glue/Athena
catalog + Lambda, added mid-phase — part of the same cloud-infrastructure
requirement bucket as S3/Athena in `ARCHITECTURE.md`'s requirement
coverage map). **This is still a partial phase completion** — labeled 🟡
in `README.md`'s phase table, not ✅. What's built and what's still open
are both listed explicitly below.

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

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge                              # strict, 52 source files, 0 errors
uv run mypy infra/lambda/ingest_trigger/handler.py  # strict, 0 errors
uv run pytest -q                                     # 164 passed, 90.33% coverage (threshold 80%)
```

Plus: all 6 `.tf` files (including the new `glue_athena.tf`) parsed
successfully with `python-hcl2`. Manually ran the full `forge ingest` →
`detect2d --mode train` → `detect2d --mode infer --local` sequence
end-to-end after the `run_inference` refactor to confirm the sequential
path still works identically (300 detections from the fixture, same as
every prior phase's manual check). Manually ran `--distributed` through
the real CLI too, confirming the same crash signature as the isolated
Ray diagnostics — consistent evidence, not a code-path-specific issue.

## What this phase does *not* claim

- Ray is wired into `detect2d` only — detect3d/track/fuse/label/
  evaluate/curate still run single-process; none of them have a
  `--distributed` flag yet.
- Ray's actual parallel execution was never verified working in this
  development environment — only its API usage, via mocks.
- The Glue catalog covers `pseudo_labels` only, not the other ~9 lake
  tables.
- Nothing consumes the Lambda's SQS queue yet — that's a Ray/ECS worker's
  job, which doesn't exist.
- The Lambda and Terraform infrastructure have never been deployed or
  invoked against real AWS (`terraform apply` is never run here, by
  policy) — only unit-tested with mocked boto3.
- Terraform files are HCL-syntax-valid (`python-hcl2`), not
  `terraform validate`-checked against the real AWS provider schema (no
  `terraform` binary was available in the environment that built this).

## Known gaps carried forward

See `KNOWN_GAPS.md` for the full list and `ARCHITECTURE.md` for how this
phase maps back to the platform's requirement-coverage table.
