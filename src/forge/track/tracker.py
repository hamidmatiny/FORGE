"""SORT-style multi-object tracker: predict, associate, update, retire."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.schemas import Detection2DRecord
from forge.track.association import associate
from forge.track.kalman import BBox, KalmanBoxTracker

StepResult = tuple[Detection2DRecord, str, int]  # (detection, track_id, hit_streak)


@dataclass
class _ActiveTrack:
    track_id: str
    kalman: KalmanBoxTracker
    class_id: int
    class_name: str
    time_since_update: int = 0
    hit_streak: int = field(default=1)


class SortTracker:
    """One tracker instance per (scene, sensor) sequence — tracks never span scenes."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[_ActiveTrack] = []
        self._next_id = 0

    def _new_track_id(self) -> str:
        self._next_id += 1
        return f"track-{self._next_id:06d}"

    def step(self, detections: list[Detection2DRecord]) -> list[StepResult]:
        """Advance the tracker by one frame.

        Call this once per frame in timestamp order, even for frames with no
        detections (that still ages and retires stale tracks correctly).
        """
        predicted_boxes: list[BBox] = [t.kalman.predict() for t in self.tracks]
        detection_boxes: list[BBox] = [
            (d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]) for d in detections
        ]

        matches, unmatched_tracks, unmatched_dets = associate(
            predicted_boxes, detection_boxes, self.iou_threshold
        )

        results: list[StepResult] = []

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.kalman.update(detection_boxes[det_idx])
            track.time_since_update = 0
            track.hit_streak += 1
            track.class_id = det.class_id
            track.class_name = det.class_name
            results.append((det, track.track_id, track.hit_streak))

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = _ActiveTrack(
                track_id=self._new_track_id(),
                kalman=KalmanBoxTracker(detection_boxes[det_idx]),
                class_id=det.class_id,
                class_name=det.class_name,
            )
            self.tracks.append(new_track)
            results.append((det, new_track.track_id, new_track.hit_streak))

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return results
