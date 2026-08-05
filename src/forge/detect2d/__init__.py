"""2D object detection: model, training loop, and inference."""

from forge.detect2d.infer import load_detector, run_inference
from forge.detect2d.model import CLASS_NAMES, Detector2DModule
from forge.detect2d.train import train_detector

__all__ = [
    "CLASS_NAMES",
    "Detector2DModule",
    "load_detector",
    "run_inference",
    "train_detector",
]
