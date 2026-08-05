"""Trust scoring and active-learning review priority.

Two ideas, both textbook active learning, applied to fused detections
rather than raw model logits:

1. **Cross-modal agreement as a confidence signal.** An object seen by
   both the camera and lidar detectors independently is more trustworthy
   than one seen by only one — this is "query by committee" in spirit
   (agreement between independent estimators), just with two sensor
   modalities standing in for committee members instead of an ensemble
   of models.
2. **Entropy-based review priority.** Objects whose trust score sits near
   the decision boundary (neither clearly good nor clearly bad) carry the
   most information if a human reviews them — the same intuition behind
   least-confidence / entropy sampling in classical active learning.
"""

from __future__ import annotations

import math

# Discount applied to single-modality detections (camera_only or
# lidar_only): they lack the cross-modal confirmation a matched detection
# gets, so their raw detector confidence is treated as less trustworthy.
# A heuristic, not a derived constant -- documented in DECISIONS.md.
DEFAULT_SINGLE_MODALITY_DISCOUNT = 0.7


def compute_trust_score(
    fusion_type: str,
    score_2d: float | None,
    score_3d: float | None,
    single_modality_discount: float = DEFAULT_SINGLE_MODALITY_DISCOUNT,
) -> float:
    """Combine per-modality detector confidence into one trust score in [0, 1].

    Args:
        fusion_type: 'matched', 'camera_only', or 'lidar_only'.
        score_2d: The camera detection's confidence, if this row has one.
        score_3d: The lidar detection's confidence, if this row has one.
        single_modality_discount: Multiplier applied when only one modality
            observed the object (see module docstring).

    Raises:
        ValueError: If a required score is missing for the given fusion_type.
    """
    if fusion_type == "matched":
        if score_2d is None or score_3d is None:
            raise ValueError("fusion_type='matched' requires both score_2d and score_3d.")
        trust = (score_2d + score_3d) / 2.0
    elif fusion_type == "camera_only":
        if score_2d is None:
            raise ValueError("fusion_type='camera_only' requires score_2d.")
        trust = score_2d * single_modality_discount
    elif fusion_type == "lidar_only":
        if score_3d is None:
            raise ValueError("fusion_type='lidar_only' requires score_3d.")
        trust = score_3d * single_modality_discount
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type!r}")

    return max(0.0, min(1.0, trust))


def binary_entropy(probability: float) -> float:
    """Shannon entropy (bits) of a Bernoulli variable with the given probability.

    Maximized at ``probability=0.5`` (maximum uncertainty) and 0 at the
    extremes (0 or 1, fully certain either way).
    """
    p = min(max(probability, 1e-6), 1.0 - 1e-6)  # avoid log(0)
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
