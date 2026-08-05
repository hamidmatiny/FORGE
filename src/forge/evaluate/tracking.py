"""Log evaluation metrics to MLflow and Weights & Biases.

Both run fully local/offline — no network calls, no API keys, no hosted
server:
- MLflow uses a local SQLite tracking store (``mlflow/mlflow.db`` under
  the data lake root). MLflow's plain filesystem store is now
  maintenance-mode-only in current versions; SQLite is the supported
  local-only alternative.
- W&B runs with ``mode="offline"``, writing run data to a local directory
  instead of syncing to the cloud. A user with a real W&B account can
  later run ``wandb sync <dir>`` to upload it — never done automatically
  here.

Failures in either backend are logged as warnings, not raised — losing
experiment-tracking metadata shouldn't fail an otherwise-successful
evaluation run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.logging import configure_logging, get_logger

_LOGGER_NAME = "forge.evaluate.tracking"


def log_to_mlflow(
    lake_root: Path,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log a run to a local MLflow SQLite store under ``<lake_root>/mlflow/``."""
    configure_logging()
    logger = get_logger(_LOGGER_NAME)
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_not_installed", note="Skipping MLflow logging.")
        return

    mlflow_dir = lake_root / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    try:
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
        mlflow.set_experiment("forge-evaluate")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as exc:  # pragma: no cover - defensive, backend-specific failures
        logger.warning("mlflow_logging_failed", error=str(exc))


def log_to_wandb(
    lake_root: Path,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log a run to a local offline W&B directory under ``<lake_root>/wandb/``."""
    configure_logging()
    logger = get_logger(_LOGGER_NAME)
    try:
        import wandb
    except ImportError:
        logger.warning("wandb_not_installed", note="Skipping W&B logging.")
        return

    wandb_dir = lake_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    try:
        run = wandb.init(
            project="forge-evaluate",
            name=run_name,
            mode="offline",
            dir=str(wandb_dir),
            config=params,
        )
        wandb.log(metrics)
        run.finish()
    except Exception as exc:  # pragma: no cover - defensive, backend-specific failures
        logger.warning("wandb_logging_failed", error=str(exc))
