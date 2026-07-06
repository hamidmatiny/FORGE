"""Structured logging configuration for FORGE."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, cast

import structlog
from structlog.typing import FilteringBoundLogger

from forge import __version__


def _get_git_sha() -> str:
    """Return short git SHA or 'unknown' when not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output to stderr."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a bound structlog logger."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))


def log_cli_invocation(
    command: str,
    args: dict[str, Any],
    log_level: str = "INFO",
) -> FilteringBoundLogger:
    """Configure logging and emit standard CLI invocation metadata."""
    configure_logging(log_level)
    logger = get_logger("forge.cli")
    logger.info(
        "cli_invocation",
        command=command,
        args=args,
        version=__version__,
        git_sha=_get_git_sha(),
    )
    return logger
