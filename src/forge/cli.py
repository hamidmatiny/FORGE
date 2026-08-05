"""FORGE CLI — single entry point for all pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from forge import __version__
from forge.config import load_config, to_container
from forge.ingest import ingest_nuscenes
from forge.logging import log_cli_invocation
from forge.settings import get_settings

app = typer.Typer(
    name="forge",
    help="FORGE — Fleet Offline Recognition & Ground-truth Engine",
    no_args_is_help=True,
)
console = Console(stderr=True)

KNOWN_GAPS_PATH = Path("KNOWN_GAPS.md")

# Phase assignments for commands not yet implemented.
PHASE_MAP: dict[str, int] = {
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
    input_dir: Annotated[
        Path,
        typer.Option("--input-dir", help="nuScenes root dir (contains <version>/*.json)."),
    ],
    version: Annotated[
        str, typer.Option("--nuscenes-version", help="nuScenes version directory name.")
    ] = "v1.0-mini",
    split: Annotated[str, typer.Option("--split", help="Dataset split label to record.")] = (
        "mini_train"
    ),
    all_sweeps: Annotated[
        bool, typer.Option("--all-sweeps", help="Ingest non-keyframe sweeps too (Phase 1: off).")
    ] = False,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Ingest a nuScenes-devkit-format dataset into the versioned Parquet data lake."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="ingest",
        args={
            "input_dir": str(input_dir),
            "version": version,
            "split": split,
            "all_sweeps": all_sweeps,
            "local": local,
        },
        log_level=settings.log_level,
    )

    cfg = to_container(
        load_config(
            "ingest",
            overrides=[
                f"input_dir={input_dir}",
                f"version={version}",
                f"split={split}",
                f"key_frames_only={not all_sweeps}",
            ],
        )
    )

    try:
        result = ingest_nuscenes(
            input_dir=Path(str(cfg["input_dir"])),
            lake_root=settings.data_lake_root,
            version=str(cfg["version"]),
            split=str(cfg["split"]),
            key_frames_only=bool(cfg["key_frames_only"]),
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]OK[/green] wrote {result.frames_written} frames, "
        f"{result.calibration_written} calibration rows, "
        f"{result.ego_poses_written} ego-pose rows "
        f"across {result.scenes_seen} scene(s) -> {result.output_dir}",
        highlight=False,
    )


@app.command("detect2d")
def detect2d_cmd(
    mode: Annotated[str, typer.Option("--mode", help="'train' or 'infer'.")] = "infer",
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Model checkpoint to load (infer mode)."),
    ] = None,
    output_checkpoint: Annotated[
        Path,
        typer.Option("--output-checkpoint", help="Where to save the checkpoint (train mode)."),
    ] = Path("checkpoints/detect2d.pt"),
    images_root: Annotated[
        Path | None,
        typer.Option(
            "--images-root", help="Dataset root for resolving frame.data_path (infer mode)."
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Training steps (train mode).")
    ] = 10,
    score_threshold: Annotated[
        float, typer.Option("--score-threshold", help="Minimum score to keep (infer mode).")
    ] = 0.3,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Train or run 2D object detection on camera frames (Faster R-CNN + Lightning)."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="detect2d",
        args={
            "mode": mode,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "images_root": str(images_root) if images_root else None,
            "max_steps": max_steps,
            "score_threshold": score_threshold,
            "local": local,
        },
        log_level=settings.log_level,
    )

    if mode not in ("train", "infer"):
        console.print(f"[red]Error:[/red] Unknown --mode '{mode}'. Use 'train' or 'infer'.")
        raise typer.Exit(code=1)

    if mode == "infer" and images_root is None:
        console.print(
            "[red]Error:[/red] --images-root is required for --mode infer.", highlight=False
        )
        raise typer.Exit(code=1)

    try:
        from forge.detect2d import load_detector, run_inference, train_detector
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge detect2d requires the [detect2d] extra "
            "(torch, torchvision, lightning). Install with: uv sync --extra detect2d",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    if mode == "train":
        final_loss = train_detector(output_checkpoint=output_checkpoint, max_steps=max_steps)
        console.print(
            f"[green]OK[/green] trained {max_steps} steps, final loss={final_loss:.4f} "
            f"-> {output_checkpoint}",
            highlight=False,
        )
        return

    assert images_root is not None  # validated above
    from forge.schemas import Detections2DTable, FramesTable

    frames_path = settings.data_lake_root / "frames.parquet"
    if not frames_path.exists():
        console.print(
            f"[red]Error:[/red] {frames_path} not found. Run 'forge ingest' first.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    frames = FramesTable.read_parquet(str(frames_path))
    model, model_version = load_detector(checkpoint)
    detections = run_inference(
        frames, images_root, model, model_version, score_threshold=score_threshold
    )
    output_path = settings.data_lake_root / "detections_2d.parquet"
    Detections2DTable.write_parquet(detections, str(output_path))
    console.print(
        f"[green]OK[/green] wrote {len(detections)} detections from "
        f"{len(frames)} lake frames (model={model_version}) -> {output_path}",
        highlight=False,
    )


@app.command("detect3d")
def detect3d_cmd(
    mode: Annotated[str, typer.Option("--mode", help="'train' or 'infer'.")] = "infer",
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Model checkpoint to load (infer mode)."),
    ] = None,
    output_checkpoint: Annotated[
        Path,
        typer.Option("--output-checkpoint", help="Where to save the checkpoint (train mode)."),
    ] = Path("checkpoints/detect3d.pt"),
    pointcloud_root: Annotated[
        Path | None,
        typer.Option(
            "--pointcloud-root",
            help="Dataset root for resolving frame.data_path (infer mode).",
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Training steps (train mode).")
    ] = 10,
    score_threshold: Annotated[
        float, typer.Option("--score-threshold", help="Minimum score to keep (infer mode).")
    ] = 0.3,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Train or run 3D object detection on lidar frames (PointNet-style + Lightning)."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="detect3d",
        args={
            "mode": mode,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "pointcloud_root": str(pointcloud_root) if pointcloud_root else None,
            "max_steps": max_steps,
            "score_threshold": score_threshold,
            "local": local,
        },
        log_level=settings.log_level,
    )

    if mode not in ("train", "infer"):
        console.print(f"[red]Error:[/red] Unknown --mode '{mode}'. Use 'train' or 'infer'.")
        raise typer.Exit(code=1)

    if mode == "infer" and pointcloud_root is None:
        console.print(
            "[red]Error:[/red] --pointcloud-root is required for --mode infer.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    try:
        from forge.detect3d import load_detector, run_inference, train_detector
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge detect3d requires the [detect3d] extra "
            "(torch, lightning, numpy). Install with: uv sync --extra detect3d",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    if mode == "train":
        final_loss = train_detector(output_checkpoint=output_checkpoint, max_steps=max_steps)
        console.print(
            f"[green]OK[/green] trained {max_steps} steps, final loss={final_loss:.4f} "
            f"-> {output_checkpoint}",
            highlight=False,
        )
        return

    assert pointcloud_root is not None  # validated above
    from forge.schemas import Detections3DTable, FramesTable

    frames_path = settings.data_lake_root / "frames.parquet"
    if not frames_path.exists():
        console.print(
            f"[red]Error:[/red] {frames_path} not found. Run 'forge ingest' first.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    frames = FramesTable.read_parquet(str(frames_path))
    model, model_version = load_detector(checkpoint)
    detections = run_inference(
        frames, pointcloud_root, model, model_version, score_threshold=score_threshold
    )
    output_path = settings.data_lake_root / "detections_3d.parquet"
    Detections3DTable.write_parquet(detections, str(output_path))
    console.print(
        f"[green]OK[/green] wrote {len(detections)} detections from "
        f"{len(frames)} lake frames (model={model_version}) -> {output_path}",
        highlight=False,
    )


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
