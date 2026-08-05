"""Active learning + pseudo-labeling: trust scoring and confidence-gated routing."""

from forge.label.run import LABELER_VERSION, run_labeling
from forge.label.scoring import (
    DEFAULT_SINGLE_MODALITY_DISCOUNT,
    binary_entropy,
    compute_trust_score,
)

__all__ = [
    "DEFAULT_SINGLE_MODALITY_DISCOUNT",
    "LABELER_VERSION",
    "binary_entropy",
    "compute_trust_score",
    "run_labeling",
]
