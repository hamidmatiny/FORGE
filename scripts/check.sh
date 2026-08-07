#!/usr/bin/env bash
# Full verification suite, in the right order, with all extras installed
# correctly every time. Exists because "run mypy/pytest with only some
# extras installed" has caused confusing (but expected) failures multiple
# times across this project's history — see KNOWN_GAPS.md's "Partial-extras
# mypy/pytest runs" row. Run this instead of piecing the commands together
# by hand.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> uv sync --all-extras --dev"
uv sync --all-extras --dev

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> mypy src/forge"
uv run mypy src/forge

echo "==> mypy infra/lambda/ingest_trigger/handler.py"
uv run mypy infra/lambda/ingest_trigger/handler.py

echo "==> pytest"
uv run pytest -q

echo "==> Terraform HCL syntax (python-hcl2 -- NOT the 'hcl2' or 'hcl' PyPI packages, both wrong)"
python3 -c "import hcl2" 2>/dev/null || pip install python-hcl2 --quiet
python3 - << 'PYEOF'
import glob
import sys

import hcl2

failed = False
for path in sorted(glob.glob("infra/terraform/*.tf")):
    try:
        with open(path) as f:
            hcl2.load(f)
        print(f"{path} OK")
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        print(f"{path} ERROR: {exc}")
        failed = True
sys.exit(1 if failed else 0)
PYEOF

echo "==> Step Functions state machine structural validation"
python3 scripts/validate_state_machine.py

echo "==> Glue catalog columns match real PyArrow schemas"
uv run python3 scripts/validate_glue_schemas.py

echo
echo "All checks passed."
