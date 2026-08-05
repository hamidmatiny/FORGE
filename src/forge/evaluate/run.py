"""Run evaluation: score pseudo-labels against ground truth, per class and overall."""

from __future__ import annotations

import uuid
from collections import defaultdict

from forge.evaluate.metrics import EvalResult, evaluate_class
from forge.schemas import EvalMetricRecord, GroundTruthRecord, PseudoLabelRecord

EVAL_VERSION = "bev-distance-ap-v1"

# Only rows with a real 3D center can be meaningfully compared to nuScenes'
# 3D ground truth -- camera_only pseudo-labels have center_xyz=[0,0,0] as a
# sentinel (see fused_objects/pseudo_labels schemas) and would corrupt the
# distance-based matching if included.
_3D_GROUNDED_FUSION_TYPES = {"matched", "lidar_only"}


def run_evaluation(
    pseudo_labels: list[PseudoLabelRecord],
    ground_truth: list[GroundTruthRecord],
    decision_filter: str = "auto_accept",
    distance_threshold_m: float = 2.0,
) -> list[EvalMetricRecord]:
    """Score pseudo-labels against ground truth, one row per class plus one 'overall' row.

    Args:
        pseudo_labels: Rows from ``forge label``.
        ground_truth: Rows from nuScenes GT (eval-only, see README.md).
        decision_filter: Which ``pseudo_labels.decision`` value to evaluate
            (default ``auto_accept`` — "how good are the labels we'd
            actually use"). Pass ``"all"`` to evaluate every decision.
        distance_threshold_m: BEV center-distance match threshold.
    """
    eval_run_id = str(uuid.uuid4())

    candidates = [
        p
        for p in pseudo_labels
        if p.fusion_type in _3D_GROUNDED_FUSION_TYPES
        and (decision_filter == "all" or p.decision == decision_filter)
    ]

    predictions_by_class: dict[str, list[tuple[tuple[float, float], float]]] = defaultdict(list)
    for prediction in candidates:
        point = (prediction.center_xyz[0], prediction.center_xyz[1])
        predictions_by_class[prediction.class_name].append((point, prediction.trust_score))

    gt_by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for gt in ground_truth:
        gt_by_class[gt.category_name].append((gt.center_xyz[0], gt.center_xyz[1]))

    all_classes = sorted(set(predictions_by_class) | set(gt_by_class))

    records: list[EvalMetricRecord] = []
    per_class_results: list[EvalResult] = []

    for class_name in all_classes:
        class_predictions = predictions_by_class.get(class_name, [])
        class_gt = gt_by_class.get(class_name, [])

        result = evaluate_class(class_predictions, class_gt, distance_threshold_m)
        per_class_results.append(result)
        records.append(_to_record(eval_run_id, class_name, result, distance_threshold_m))

    records.append(
        _to_record(eval_run_id, "overall", _aggregate(per_class_results), distance_threshold_m)
    )

    return records


def _aggregate(per_class_results: list[EvalResult]) -> EvalResult:
    """Combine per-class results: micro-averaged precision/recall/F1, mean AP (mAP).

    Matches don't get pooled across classes for AP itself -- a car can
    never "match" a pedestrian GT box here, unlike a naive class-agnostic
    pooling would allow. mAP (mean of per-class AP) is the standard
    detection-benchmark convention for a single headline number.
    """
    if not per_class_results:
        return EvalResult(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    num_gt = sum(r.num_gt for r in per_class_results)
    num_predictions = sum(r.num_predictions for r in per_class_results)
    num_matched = sum(r.num_matched for r in per_class_results)
    precision = num_matched / num_predictions if num_predictions else 0.0
    recall = num_matched / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_ap = sum(r.average_precision for r in per_class_results) / len(per_class_results)

    return EvalResult(
        num_gt=num_gt,
        num_predictions=num_predictions,
        num_matched=num_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=mean_ap,
    )


def _to_record(
    eval_run_id: str, class_name: str, result: EvalResult, distance_threshold_m: float
) -> EvalMetricRecord:
    return EvalMetricRecord(
        eval_run_id=eval_run_id,
        class_name=class_name,
        num_gt=result.num_gt,
        num_predictions=result.num_predictions,
        num_matched=result.num_matched,
        precision=result.precision,
        recall=result.recall,
        f1=result.f1,
        average_precision=result.average_precision,
        distance_threshold_m=distance_threshold_m,
        eval_version=EVAL_VERSION,
    )
