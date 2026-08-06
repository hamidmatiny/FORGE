"""FORGE CLI — single entry point for all pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from forge import __version__
from forge.config import load_config, to_container
from forge.ingest import ingest_nuscenes
from forge.label.scoring import DEFAULT_SINGLE_MODALITY_DISCOUNT
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
    distributed: Annotated[
        bool,
        typer.Option(
            "--distributed", help="Run inference across local CPU cores via Ray (infer mode)."
        ),
    ] = False,
) -> None:
    """Train or run 2D object detection on camera frames (Faster R-CNN + Lightning)."""
    if not local and not distributed:
        console.print(
            "[red]Error:[/red] Pass --local (single-process) or --distributed (local Ray).",
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
            "distributed": distributed,
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
            "[red]Error:[/red] forge detect2d requires the 'detect2d' extra "
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
        frames,
        images_root,
        model,
        model_version,
        score_threshold=score_threshold,
        distributed=distributed,
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
            "[red]Error:[/red] forge detect3d requires the 'detect3d' extra "
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
    iou_threshold: Annotated[
        float, typer.Option("--iou-threshold", help="Minimum IoU to match a detection to a track.")
    ] = 0.3,
    max_age: Annotated[
        int,
        typer.Option("--max-age", help="Frames a track survives with no matching detection."),
    ] = 3,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Associate 2D detections across frames into tracks (SORT: Kalman filter + Hungarian IoU)."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="track",
        args={"iou_threshold": iou_threshold, "max_age": max_age, "local": local},
        log_level=settings.log_level,
    )

    try:
        from forge.track import run_tracking
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge track requires the 'track' extra (numpy, scipy). "
            "Install with: uv sync --extra track",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    from forge.schemas import Detections2DTable, FramesTable, TracksTable

    frames_path = settings.data_lake_root / "frames.parquet"
    detections_path = settings.data_lake_root / "detections_2d.parquet"
    if not frames_path.exists():
        console.print(
            f"[red]Error:[/red] {frames_path} not found. Run 'forge ingest' first.",
            highlight=False,
        )
        raise typer.Exit(code=1)
    if not detections_path.exists():
        console.print(
            f"[red]Error:[/red] {detections_path} not found. Run 'forge detect2d' first.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    frames = FramesTable.read_parquet(str(frames_path))
    detections = Detections2DTable.read_parquet(str(detections_path))
    tracks = run_tracking(detections, frames, iou_threshold=iou_threshold, max_age=max_age)

    output_path = settings.data_lake_root / "tracks.parquet"
    TracksTable.write_parquet(tracks, str(output_path))
    num_unique_tracks = len({t.track_id for t in tracks})
    console.print(
        f"[green]OK[/green] wrote {len(tracks)} track rows ({num_unique_tracks} unique tracks) "
        f"from {len(detections)} detections -> {output_path}",
        highlight=False,
    )


@app.command()
def fuse(
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou-threshold", help="Minimum IoU to match a projected lidar box to a camera box."
        ),
    ] = 0.1,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Fuse camera and lidar detections via calibrated projection + IoU association."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="fuse",
        args={"iou_threshold": iou_threshold, "local": local},
        log_level=settings.log_level,
    )

    try:
        from forge.fuse import run_fusion
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge fuse requires the 'fuse' extra (numpy, scipy). "
            "Install with: uv sync --extra fuse",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    from forge.schemas import (
        CalibrationTable,
        Detections2DTable,
        Detections3DTable,
        FramesTable,
        FusedObjectsTable,
    )

    required = {
        "frames.parquet": "forge ingest",
        "calibration.parquet": "forge ingest",
        "detections_2d.parquet": "forge detect2d --mode infer",
        "detections_3d.parquet": "forge detect3d --mode infer",
    }
    for filename, command_hint in required.items():
        path = settings.data_lake_root / filename
        if not path.exists():
            console.print(
                f"[red]Error:[/red] {path} not found. Run '{command_hint}' first.",
                highlight=False,
            )
            raise typer.Exit(code=1)

    frames = FramesTable.read_parquet(str(settings.data_lake_root / "frames.parquet"))
    calibration = CalibrationTable.read_parquet(
        str(settings.data_lake_root / "calibration.parquet")
    )
    detections_2d = Detections2DTable.read_parquet(
        str(settings.data_lake_root / "detections_2d.parquet")
    )
    detections_3d = Detections3DTable.read_parquet(
        str(settings.data_lake_root / "detections_3d.parquet")
    )

    fused = run_fusion(
        detections_2d, detections_3d, frames, calibration, iou_threshold=iou_threshold
    )

    output_path = settings.data_lake_root / "fused_objects.parquet"
    FusedObjectsTable.write_parquet(fused, str(output_path))

    matched = sum(1 for f in fused if f.fusion_type == "matched")
    camera_only = sum(1 for f in fused if f.fusion_type == "camera_only")
    lidar_only = sum(1 for f in fused if f.fusion_type == "lidar_only")
    console.print(
        f"[green]OK[/green] wrote {len(fused)} fused objects "
        f"({matched} matched, {camera_only} camera-only, {lidar_only} lidar-only) -> {output_path}",
        highlight=False,
    )


@app.command()
def label(
    auto_accept_threshold: Annotated[
        float,
        typer.Option(
            "--auto-accept-threshold", help="Trust score at or above which a label needs no review."
        ),
    ] = 0.7,
    reject_threshold: Annotated[
        float,
        typer.Option("--reject-threshold", help="Trust score below which a label is rejected."),
    ] = 0.3,
    single_modality_discount: Annotated[
        float,
        typer.Option(
            "--single-modality-discount",
            help="Multiplier for camera_only/lidar_only trust (no cross-modal confirmation).",
        ),
    ] = DEFAULT_SINGLE_MODALITY_DISCOUNT,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Active-learning selection + pseudo-label generation with confidence-gated review queue."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="label",
        args={
            "auto_accept_threshold": auto_accept_threshold,
            "reject_threshold": reject_threshold,
            "single_modality_discount": single_modality_discount,
            "local": local,
        },
        log_level=settings.log_level,
    )

    from forge.label import run_labeling
    from forge.schemas import (
        Detections2DTable,
        Detections3DTable,
        FusedObjectsTable,
        PseudoLabelsTable,
    )

    fused_path = settings.data_lake_root / "fused_objects.parquet"
    if not fused_path.exists():
        console.print(
            f"[red]Error:[/red] {fused_path} not found. Run 'forge fuse' first.", highlight=False
        )
        raise typer.Exit(code=1)

    fused_objects = FusedObjectsTable.read_parquet(str(fused_path))
    detections_2d = Detections2DTable.read_parquet(
        str(settings.data_lake_root / "detections_2d.parquet")
    )
    detections_3d = Detections3DTable.read_parquet(
        str(settings.data_lake_root / "detections_3d.parquet")
    )

    try:
        labels = run_labeling(
            fused_objects,
            detections_2d,
            detections_3d,
            auto_accept_threshold=auto_accept_threshold,
            reject_threshold=reject_threshold,
            single_modality_discount=single_modality_discount,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc

    output_path = settings.data_lake_root / "pseudo_labels.parquet"
    PseudoLabelsTable.write_parquet(labels, str(output_path))

    auto_accept = sum(1 for label_ in labels if label_.decision == "auto_accept")
    needs_review = sum(1 for label_ in labels if label_.decision == "needs_review")
    rejected = sum(1 for label_ in labels if label_.decision == "rejected")
    console.print(
        f"[green]OK[/green] wrote {len(labels)} pseudo-labels "
        f"({auto_accept} auto-accept, {needs_review} needs-review, {rejected} rejected) "
        f"-> {output_path}",
        highlight=False,
    )


@app.command()
def evaluate(
    gt_input_dir: Annotated[
        Path,
        typer.Option(
            "--gt-input-dir",
            help="nuScenes root with sample_annotation.json (eval-only, never a pipeline input).",
        ),
    ],
    version: Annotated[
        str, typer.Option("--nuscenes-version", help="nuScenes version directory name.")
    ] = "v1.0-mini",
    decision_filter: Annotated[
        str,
        typer.Option(
            "--decision-filter",
            help="Which pseudo_labels.decision to evaluate: auto_accept/needs_review/all.",
        ),
    ] = "auto_accept",
    distance_threshold: Annotated[
        float,
        typer.Option("--distance-threshold", help="BEV center-distance match threshold (meters)."),
    ] = 2.0,
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Evaluate auto-labels against ground truth and log quality metrics (evaluation only)."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="evaluate",
        args={
            "gt_input_dir": str(gt_input_dir),
            "version": version,
            "decision_filter": decision_filter,
            "distance_threshold": distance_threshold,
            "local": local,
        },
        log_level=settings.log_level,
    )

    labels_path = settings.data_lake_root / "pseudo_labels.parquet"
    if not labels_path.exists():
        console.print(
            f"[red]Error:[/red] {labels_path} not found. Run 'forge label' first.", highlight=False
        )
        raise typer.Exit(code=1)

    try:
        from forge.evaluate import ingest_ground_truth, log_to_mlflow, log_to_wandb, run_evaluation
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge evaluate requires the 'evaluate' extra "
            "(mlflow-skinny, wandb). Install with: uv sync --extra evaluate",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    from forge.schemas import EvalMetricsTable, PseudoLabelsTable

    try:
        ground_truth = ingest_ground_truth(gt_input_dir, version=version)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from exc

    pseudo_labels = PseudoLabelsTable.read_parquet(str(labels_path))
    metrics = run_evaluation(
        pseudo_labels,
        ground_truth,
        decision_filter=decision_filter,
        distance_threshold_m=distance_threshold,
    )

    output_path = settings.data_lake_root / "eval_metrics.parquet"
    EvalMetricsTable.write_parquet(metrics, str(output_path))

    overall = next(m for m in metrics if m.class_name == "overall")
    run_params = {
        "decision_filter": decision_filter,
        "distance_threshold_m": distance_threshold,
        "gt_input_dir": str(gt_input_dir),
    }
    run_metrics = {
        "precision": overall.precision,
        "recall": overall.recall,
        "f1": overall.f1,
        "mAP": overall.average_precision,
    }
    log_to_mlflow(settings.data_lake_root, "forge-evaluate", run_params, run_metrics)
    log_to_wandb(settings.data_lake_root, "forge-evaluate", run_params, run_metrics)

    console.print(
        f"[green]OK[/green] evaluated {overall.num_predictions} predictions against "
        f"{overall.num_gt} GT boxes -> precision={overall.precision:.3f} "
        f"recall={overall.recall:.3f} f1={overall.f1:.3f} mAP={overall.average_precision:.3f} "
        f"-> {output_path}",
        highlight=False,
    )


@app.command()
def curate(
    distance_threshold: Annotated[
        float,
        typer.Option(
            "--distance-threshold", help="Max feature-space distance to count as a near-duplicate."
        ),
    ] = 1.0,
    decision_filter: Annotated[
        str,
        typer.Option("--decision-filter", help="Which pseudo_labels.decision to curate."),
    ] = "auto_accept",
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Curate, deduplicate (LanceDB vector search), and export annotation datasets."""
    if not local:
        console.print(
            "[red]Error:[/red] Distributed (Ray) execution lands in Phase 9. Pass --local for now.",
            highlight=False,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_cli_invocation(
        command="curate",
        args={
            "distance_threshold": distance_threshold,
            "decision_filter": decision_filter,
            "local": local,
        },
        log_level=settings.log_level,
    )

    labels_path = settings.data_lake_root / "pseudo_labels.parquet"
    if not labels_path.exists():
        console.print(
            f"[red]Error:[/red] {labels_path} not found. Run 'forge label' first.", highlight=False
        )
        raise typer.Exit(code=1)

    try:
        from forge.curate import run_curation
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] forge curate requires the 'curate' extra (lancedb). "
            "Install with: uv sync --extra curate",
            highlight=False,
        )
        raise typer.Exit(code=1) from exc

    from forge.schemas import CuratedTable, PseudoLabelsTable

    pseudo_labels = PseudoLabelsTable.read_parquet(str(labels_path))
    lancedb_path = settings.data_lake_root / "lancedb"
    curated = run_curation(
        pseudo_labels,
        lancedb_path,
        distance_threshold=distance_threshold,
        decision_filter=decision_filter,
    )

    output_path = settings.data_lake_root / "curated.parquet"
    CuratedTable.write_parquet(curated, str(output_path))

    num_kept = sum(1 for c in curated if not c.is_duplicate)
    num_duplicates = sum(1 for c in curated if c.is_duplicate)
    console.print(
        f"[green]OK[/green] curated {len(curated)} labels ({num_kept} kept, "
        f"{num_duplicates} near-duplicates) -> {output_path}",
        highlight=False,
    )


@app.command()
def visualize(
    local: Annotated[bool, typer.Option("--local", help="Run in single-process mode.")] = False,
) -> None:
    """Launch interactive review (rerun.io / Foxglove MCAP / FiftyOne)."""
    _not_implemented("visualize", local=local)


if __name__ == "__main__":
    app()
