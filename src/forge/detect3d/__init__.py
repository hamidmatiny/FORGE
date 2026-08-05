"""3D object detection: point-cloud model, training loop, and inference."""

from forge.detect3d.infer import load_detector, run_inference
from forge.detect3d.model import CLASS_NAMES, Detector3DModule
from forge.detect3d.train import train_detector

__all__ = [
    "CLASS_NAMES",
    "Detector3DModule",
    "load_detector",
    "run_inference",
    "train_detector",
]
