"""Detection evaluation metrics: BEV center-distance matching + precision/recall/AP.

Uses BEV (bird's-eye-view) center-distance matching rather than 3D IoU —
this is deliberately the same convention nuScenes' own official detection
metric uses (distance thresholds, not IoU), since it's simpler to get
right than 3D IoU and is a faithful choice for this dataset rather than an
arbitrary simplification.

Average precision follows the standard VOC2012-style all-points
interpolated precision-recall curve, computed via a score-ordered greedy
match (highest-confidence prediction gets first claim on the closest
unmatched ground-truth box within the distance threshold) — the same
methodology COCO/VOC/nuScenes detection AP use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point2D = tuple[float, float]


@dataclass(frozen=True)
class EvalResult:
    """Precision/recall/AP for one class, using every given prediction."""

    num_gt: int
    num_predictions: int
    num_matched: int
    precision: float
    recall: float
    f1: float
    average_precision: float


def _euclidean(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _average_precision(precisions: list[float], recalls: list[float]) -> float:
    """VOC2012-style all-points interpolated average precision."""
    padded_recalls = [0.0, *recalls, 1.0]
    padded_precisions = [0.0, *precisions, 0.0]

    # Precision envelope: make precision monotonically non-increasing
    # when read right-to-left (the standard interpolation step).
    for i in range(len(padded_precisions) - 2, -1, -1):
        padded_precisions[i] = max(padded_precisions[i], padded_precisions[i + 1])

    ap = 0.0
    for i in range(1, len(padded_recalls)):
        if padded_recalls[i] != padded_recalls[i - 1]:
            ap += (padded_recalls[i] - padded_recalls[i - 1]) * padded_precisions[i]
    return ap


def evaluate_class(
    predictions: list[tuple[Point2D, float]],
    ground_truth: list[Point2D],
    distance_threshold_m: float = 2.0,
) -> EvalResult:
    """Score one class's predictions against its ground-truth boxes.

    Args:
        predictions: ``(bev_center, confidence_score)`` pairs.
        ground_truth: BEV centers of the ground-truth boxes for this class.
        distance_threshold_m: Maximum BEV center distance (meters) to
            count as a match — nuScenes commonly sweeps {0.5, 1, 2, 4}.
    """
    sorted_predictions = sorted(predictions, key=lambda p: p[1], reverse=True)
    gt_matched = [False] * len(ground_truth)

    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for point, _score in sorted_predictions:
        best_distance = math.inf
        best_index = -1
        for i, gt_point in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            distance = _euclidean(point, gt_point)
            if distance < best_distance:
                best_distance = distance
                best_index = i

        if best_index >= 0 and best_distance <= distance_threshold_m:
            gt_matched[best_index] = True
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / len(ground_truth) if ground_truth else 0.0)

    num_matched = sum(gt_matched)
    num_predictions = len(predictions)
    num_gt = len(ground_truth)

    final_precision = num_matched / num_predictions if num_predictions else 0.0
    final_recall = num_matched / num_gt if num_gt else 0.0
    f1 = (
        2 * final_precision * final_recall / (final_precision + final_recall)
        if (final_precision + final_recall) > 0
        else 0.0
    )
    ap = _average_precision(precisions, recalls) if ground_truth else 0.0

    return EvalResult(
        num_gt=num_gt,
        num_predictions=num_predictions,
        num_matched=num_matched,
        precision=final_precision,
        recall=final_recall,
        f1=f1,
        average_precision=ap,
    )
