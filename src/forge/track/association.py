"""IoU computation and Hungarian assignment between tracks and detections."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from forge.track.kalman import BBox


def iou_batch(boxes_a: list[BBox], boxes_b: list[BBox]) -> npt.NDArray[np.float64]:
    """Pairwise IoU between two lists of [x1, y1, x2, y2] boxes -> (len(a), len(b))."""
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.array(boxes_a, dtype=np.float64)[:, None, :]  # (A, 1, 4)
    b = np.array(boxes_b, dtype=np.float64)[None, :, :]  # (1, B, 4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    intersection = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], a_min=0, a_max=None) * np.clip(
        a[..., 3] - a[..., 1], a_min=0, a_max=None
    )
    area_b = np.clip(b[..., 2] - b[..., 0], a_min=0, a_max=None) * np.clip(
        b[..., 3] - b[..., 1], a_min=0, a_max=None
    )
    union = area_a + area_b - intersection

    iou: npt.NDArray[np.float64] = np.where(union > 1e-6, intersection / union, 0.0)
    return iou


def associate(
    predicted_boxes: list[BBox],
    detection_boxes: list[BBox],
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted track boxes to this frame's detection boxes by IoU.

    Returns:
        (matches, unmatched_track_indices, unmatched_detection_indices), where
        matches is a list of (track_index, detection_index) pairs.
    """
    if not predicted_boxes or not detection_boxes:
        return [], list(range(len(predicted_boxes))), list(range(len(detection_boxes)))

    iou_matrix = iou_batch(predicted_boxes, detection_boxes)
    track_indices, det_indices = linear_sum_assignment(-iou_matrix)

    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(predicted_boxes)))
    unmatched_dets = set(range(len(detection_boxes)))

    for t_idx, d_idx in zip(track_indices, det_indices, strict=True):
        if iou_matrix[t_idx, d_idx] < iou_threshold:
            continue
        matches.append((int(t_idx), int(d_idx)))
        unmatched_tracks.discard(int(t_idx))
        unmatched_dets.discard(int(d_idx))

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)
