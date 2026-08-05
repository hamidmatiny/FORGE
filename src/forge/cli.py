"""FORGE CLI — single entry point for all pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from forge import __version__
from forge.logging import log_cli_invocation
from forge.settings import get_settings

app = typer.Typer(
    name="forge",
    help="FORGE — Fleet Offline Recognition & Ground-truth Engine",
    no_args_is_help=True,
)
console = Console(stderr=True)

KNOWN_GAPS_PATH = Path("KNOWN_GAPS.md")

# Phase assignments for unimplemented commands.
PHASE_MAP: dict[str, int] = {
    "ingest": 1,
    "detect2d": 2,
    "detect3d": 3,
    "track": 4,
    "fuse": 5,
    "label": 6,
    "evaluate": 7,
    "curate": 8,
    "visualize": 10,
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"forge {__version__}")
        raise typer.Exit()


def _not_implemented(command: str, local: bool = False) -> None:
    """Print and log a clear not-implemented message, then exit nonzero."""
    phase = PHASE_MAP[command]
    settings = get_settings()
    log_cli_invocation(
        command=command,
        args={"local": local},
        log_level=settings.log_level,
    )
    message = (
        f"forge {command} is not implemented until Phase {phase}. "
        f"See {KNOWN_GAPS_PATH} for details."
    )
    console.print(f"[red]Error:[/red] {message}", highlight=False)
    raise typer.Exit(code=1)


@app.callback()
def main(
    _version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """FORGE offline perception auto-labeling platform."""


@app.command()
def ingest(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Ingest logged sensor data into the versioned Parquet data lake."""
    _not_implemented("ingest", local=local)


@app.command("detect2d")
def detect2d_cmd(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Run 2D object detection on camera frames."""
    _not_implemented("detect2d", local=local)


@app.command("detect3d")
def detect3d_cmd(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Run 3D object detection on lidar/radar data."""
    _not_implemented("detect3d", local=local)


@app.command()
def track(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Associate detections across frames into tracks."""
    _not_implemented("track", local=local)


@app.command()
def fuse(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Fuse multi-sensor detections and tracks."""
    _not_implemented("fuse", local=local)


@app.command()
def label(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Active-learning selection + pseudo-label generation with confidence-gated review queue."""
    _not_implemented("label", local=local)


@app.command()
def evaluate(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Evaluate auto-labels against ground truth and log quality metrics (evaluation only)."""
    _not_implemented("evaluate", local=local)


@app.command()
def curate(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Curate, deduplicate (LanceDB vector search), and export annotation datasets."""
    _not_implemented("curate", local=local)


@app.command()
def visualize(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Launch interactive review (rerun.io / Foxglove MCAP / FiftyOne)."""
    _not_implemented("visualize", local=local)


if __name__ == "__main__":
    app()
