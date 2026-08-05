"""2D detector: torchvision Faster R-CNN wrapped as a LightningModule.

Random-initialized (``weights=None, weights_backbone=None``) so nothing is
downloaded — Phase 2's CI cost-safety policy mirrors the sibling repos'
(no real GPU spend, and here also no multi-hundred-MB pretrained-weight
downloads). This is a real, working multi-box detector architecture
(anchor-based RPN + NMS from torchvision, not a hand-rolled stand-in) with
randomly initialized weights: correct for validating the training/inference
pipeline mechanics, not yet tuned for accuracy. See KNOWN_GAPS.md.
"""

from __future__ import annotations

import lightning as pl
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

# Index 0 is reserved for background per torchvision's detection convention.
CLASS_NAMES: list[str] = [
    "background",
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_sign",
]

DetectionBatch = tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]


def build_model(num_classes: int = len(CLASS_NAMES)) -> FasterRCNN:
    """Build a randomly initialized Faster R-CNN (MobileNetV3 + FPN backbone)."""
    model: FasterRCNN = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
    return model


class Detector2DModule(pl.LightningModule):
    """LightningModule wrapping the Faster R-CNN training/inference loop."""

    def __init__(self, num_classes: int = len(CLASS_NAMES), lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(num_classes)

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = self.model(images)
        return result

    def training_step(self, batch: DetectionBatch, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        loss_dict: dict[str, torch.Tensor] = self.model(images, targets)
        total_loss = torch.stack(list(loss_dict.values())).sum()
        self.log("train_loss", total_loss, prog_bar=True, batch_size=len(images))
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, batch_size=len(images))
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams["lr"])
