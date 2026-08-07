"""Run the active-learning / pseudo-labeling policy over every fused object."""

from __future__ import annotations

import uuid

from forge.distributed import run_distributed_map
from forge.label.scoring import (
    DEFAULT_SINGLE_MODALITY_DISCOUNT,
    binary_entropy,
    compute_trust_score,
)
from forge.schemas import Detection2DRecord, Detection3DRecord, FusedObjectRecord, PseudoLabelRecord

LABELER_VERSION = "trust-threshold-v1"


def _label_one_object(
    fused: FusedObjectRecord,
    detections_2d_by_id: dict[str, Detection2DRecord],
    detections_3d_by_id: dict[str, Detection3DRecord],
    auto_accept_threshold: float,
    reject_threshold: float,
    single_modality_discount: float,
) -> PseudoLabelRecord:
    """Score one fused object. Safe to run in a Ray worker."""
    score_2d = detections_2d_by_id[fused.detection_id_2d].score if fused.detection_id_2d else None
    score_3d = detections_3d_by_id[fused.detection_id_3d].score if fused.detection_id_3d else None

    trust = compute_trust_score(fused.fusion_type, score_2d, score_3d, single_modality_discount)

    if trust >= auto_accept_threshold:
        decision = "auto_accept"
    elif trust < reject_threshold:
        decision = "rejected"
    else:
        decision = "needs_review"

    return PseudoLabelRecord(
        pseudo_label_id=str(uuid.uuid4()),
        fusion_id=fused.fusion_id,
        scene_id=fused.scene_id,
        timestamp_us=fused.timestamp_us,
        fusion_type=fused.fusion_type,
        class_id=fused.class_id,
        class_name=fused.class_name,
        bbox_xyxy=fused.bbox_xyxy,
        center_xyz=fused.center_xyz,
        dimensions_whl=fused.dimensions_whl,
        yaw=fused.yaw,
        trust_score=trust,
        decision=decision,
        review_priority=binary_entropy(trust),
        labeler_version=LABELER_VERSION,
    )


def run_labeling(
    fused_objects: list[FusedObjectRecord],
    detections_2d: list[Detection2DRecord],
    detections_3d: list[Detection3DRecord],
    auto_accept_threshold: float = 0.7,
    reject_threshold: float = 0.3,
    single_modality_discount: float = DEFAULT_SINGLE_MODALITY_DISCOUNT,
    distributed: bool = False,
) -> list[PseudoLabelRecord]:
    """Score every fused object and route it to auto_accept / needs_review / rejected.

    Looks the original per-modality detection scores back up via
    ``fused_objects``' ``detection_id_2d``/``detection_id_3d`` fields
    (fusion is intentionally lean and doesn't duplicate them) so
    :func:`~forge.label.scoring.compute_trust_score` can weigh cross-modal
    agreement for ``matched`` rows.

    Args:
        distributed: If True, scores each fused object via local Ray (see
            ``forge.distributed.run_distributed_map``) instead of a plain
            sequential loop — every object is scored independently, so
            this is genuinely parallel. The lookup dicts are passed via
            ``shared_args`` so Ray ``ray.put()``s them once.

    Raises:
        ValueError: If ``auto_accept_threshold <= reject_threshold`` (the
            decision bands would be inverted or empty).
    """
    if auto_accept_threshold <= reject_threshold:
        raise ValueError(
            f"auto_accept_threshold ({auto_accept_threshold}) must be greater than "
            f"reject_threshold ({reject_threshold})."
        )

    detections_2d_by_id = {d.detection_id: d for d in detections_2d}
    detections_3d_by_id = {d.detection_id: d for d in detections_3d}

    return run_distributed_map(
        lambda fused, det2d_by_id, det3d_by_id: _label_one_object(
            fused,
            det2d_by_id,
            det3d_by_id,
            auto_accept_threshold,
            reject_threshold,
            single_modality_discount,
        ),
        fused_objects,
        distributed=distributed,
        shared_args=(detections_2d_by_id, detections_3d_by_id),
    )
