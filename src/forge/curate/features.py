"""Geometric feature vectors for near-duplicate search.

This is explicitly **not** a learned visual embedding — no trained
embedding model exists anywhere in this pipeline (that would need a real
CNN/transformer backbone trained on real labeled data, which is exactly
what doesn't exist yet, see PHASE_2_COMPLETION.md and PHASE_3_COMPLETION.md
for why). Instead, each pseudo-label's own geometry becomes an 8-dim
feature vector: two detections of the *same physical object* should have
very similar centers/dimensions/orientation even if their exact boxes
differ slightly, which is exactly the "near-duplicate" signal curation
needs. LanceDB provides genuine ANN search infrastructure over these
vectors regardless of what generated them.
"""

from __future__ import annotations

import math

VECTOR_DIM = 8


def build_feature_vector(
    center_xyz: list[float], dimensions_whl: list[float], yaw: float
) -> list[float]:
    """[center_xyz(3), dimensions_whl(3), sin(yaw), cos(yaw)] -> an 8-dim vector.

    Yaw is encoded as (sin, cos) rather than the raw angle so that -pi and
    +pi (the same heading) end up close in feature space instead of
    maximally far apart.
    """
    return [
        center_xyz[0],
        center_xyz[1],
        center_xyz[2],
        dimensions_whl[0],
        dimensions_whl[1],
        dimensions_whl[2],
        math.sin(yaw),
        math.cos(yaw),
    ]
