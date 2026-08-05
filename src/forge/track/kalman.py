"""A constant-velocity Kalman filter over [center_x, center_y, area, aspect_ratio].

This is the classic SORT (Simple Online and Realtime Tracking) state
representation: a 7-dim state ``[cx, cy, s, r, vcx, vcy, vs]`` where
``s = width * height`` (area) and ``r = width / height`` (aspect ratio, held
constant — objects don't usually change shape frame to frame, only position
and apparent size). Implemented directly with plain NumPy linear algebra
(no external Kalman-filter library) since the predict/update equations are
compact and this keeps the dependency footprint to numpy + scipy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

StateVector = npt.NDArray[np.float64]
BBox = tuple[float, float, float, float]

_STATE_DIM = 7
_MEAS_DIM = 4

# Constant-velocity transition: position/scale advance by their velocity;
# aspect ratio and all velocities are carried forward unchanged.
_F: npt.NDArray[np.float64] = np.eye(_STATE_DIM, dtype=np.float64)
_F[0, 4] = 1.0
_F[1, 5] = 1.0
_F[2, 6] = 1.0

# Observation model: we only ever directly measure [cx, cy, s, r].
_H: npt.NDArray[np.float64] = np.zeros((_MEAS_DIM, _STATE_DIM), dtype=np.float64)
_H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0


def bbox_to_z(bbox: BBox) -> StateVector:
    """[x1, y1, x2, y2] -> [center_x, center_y, area, aspect_ratio]."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = max(w, 0.0) * max(h, 0.0)
    r = w / h if h > 1e-6 else 1.0
    return np.array([cx, cy, s, r], dtype=np.float64)


def state_to_bbox(state: StateVector) -> BBox:
    """[center_x, center_y, area, aspect_ratio, ...] -> [x1, y1, x2, y2]."""
    cx, cy, s, r = state[0], state[1], max(state[2], 1e-6), max(state[3], 1e-6)
    w = float(np.sqrt(s * r))
    h = float(s / w) if w > 1e-6 else 0.0
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxTracker:
    """One Kalman filter instance tracking a single object's bbox over time."""

    def __init__(self, initial_bbox: BBox) -> None:
        self.x: StateVector = np.zeros(_STATE_DIM, dtype=np.float64)
        self.x[:4] = bbox_to_z(initial_bbox)

        # High initial uncertainty on the (unobserved) velocity terms.
        self.p: npt.NDArray[np.float64] = np.eye(_STATE_DIM, dtype=np.float64) * 10.0
        self.p[4:, 4:] *= 100.0

        # Process noise: small, with more allowed drift on the velocity terms.
        self.q: npt.NDArray[np.float64] = np.eye(_STATE_DIM, dtype=np.float64) * 0.01
        self.q[4:, 4:] *= 0.1

        # Measurement noise: trust detector boxes reasonably well.
        self.r: npt.NDArray[np.float64] = np.eye(_MEAS_DIM, dtype=np.float64) * 1.0

    def predict(self) -> BBox:
        """Advance the state one step and return the predicted bbox."""
        self.x = _F @ self.x
        if self.x[2] + self.x[6] <= 0:  # area must stay positive
            self.x[6] = 0.0
        self.p = _F @ self.p @ _F.T + self.q
        return state_to_bbox(self.x)

    def update(self, bbox: BBox) -> None:
        """Correct the state with an observed bbox."""
        z = bbox_to_z(bbox)
        y = z - _H @ self.x
        s = _H @ self.p @ _H.T + self.r
        k = self.p @ _H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(_STATE_DIM) - k @ _H) @ self.p

    def current_bbox(self) -> BBox:
        return state_to_bbox(self.x)
