"""3D detector: a small PointNet-style point-cloud encoder + fixed-slot head.

torchvision has no off-the-shelf 3D/point-cloud detection model (unlike
Phase 2's Faster R-CNN for 2D) and real point-cloud detectors
(PointPillars, CenterPoint, ...) live in packages (mmdetection3d,
OpenPCDet) this environment can't install. So this is a genuinely
hand-built architecture: a per-point shared-MLP encoder (the core PointNet
idea — permutation-invariant via global max-pooling) feeding a small head
that predicts a **fixed number** of box "query slots" per point cloud.

To keep training honest and simple for a smoke test (see
DECISIONS.md ADR-014), the synthetic training set always emits exactly
``NUM_QUERIES`` boxes in a fixed order, so each query slot has a direct,
unambiguous training target -- no Hungarian/bipartite matching needed.
Real multi-object 3D detection with a variable object count and proper
matching is a known follow-on (KNOWN_GAPS.md), not claimed here.
"""

from __future__ import annotations

import lightning as pl
import torch
from torch import nn

# Foreground classes only -- objectness is modeled separately from class,
# unlike detect2d's torchvision convention of a dedicated background class.
CLASS_NAMES: list[str] = ["vehicle", "pedestrian", "cyclist"]

NUM_QUERIES = 4
BOX_DIM = 7  # center_xyz(3) + dimensions_whl(3) + yaw(1)
POINT_FEATURES = 4  # x, y, z, intensity (ring index is dropped)

Point3DBatch = tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]  # clouds, classes, boxes


class PointNetEncoder(nn.Module):
    """Per-point shared MLP + global max-pool -> a single permutation-invariant feature."""

    def __init__(self, in_features: int = POINT_FEATURES, out_features: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (N, in_features) for one cloud -> (out_features,) global feature."""
        per_point = self.mlp(points)  # (N, out_features)
        global_feature, _ = per_point.max(dim=0)
        return torch.as_tensor(global_feature)


class Detector3DModule(pl.LightningModule):
    """LightningModule wrapping the PointNet-style encoder + fixed-slot detection head."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        num_queries: int = NUM_QUERIES,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.encoder = PointNetEncoder(out_features=256)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_queries * (1 + num_classes + BOX_DIM)),
        )

    def forward(self, point_clouds: list[torch.Tensor]) -> torch.Tensor:
        """point_clouds: list of (N_i, 4) tensors -> (B, num_queries, 1+num_classes+7)."""
        features = torch.stack([self.encoder(pc) for pc in point_clouds])  # (B, 256)
        raw = self.head(features)  # (B, num_queries * (1+num_classes+7))
        return torch.as_tensor(raw.view(-1, self.num_queries, 1 + self.num_classes + BOX_DIM))

    def training_step(self, batch: Point3DBatch, batch_idx: int) -> torch.Tensor:
        point_clouds, target_classes, target_boxes = batch
        predictions = self(point_clouds)

        objectness_logits = predictions[..., 0]
        class_logits = predictions[..., 1 : 1 + self.num_classes]
        box_pred = predictions[..., 1 + self.num_classes :]

        objectness_target = torch.ones_like(objectness_logits)
        objectness_loss = nn.functional.binary_cross_entropy_with_logits(
            objectness_logits, objectness_target
        )
        classification_loss = nn.functional.cross_entropy(
            class_logits.reshape(-1, self.num_classes), target_classes.reshape(-1)
        )
        box_loss = nn.functional.mse_loss(box_pred, target_boxes)

        total_loss = objectness_loss + classification_loss + box_loss
        batch_size = len(point_clouds)
        self.log("train_loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("train_objectness_loss", objectness_loss, batch_size=batch_size)
        self.log("train_classification_loss", classification_loss, batch_size=batch_size)
        self.log("train_box_loss", box_loss, batch_size=batch_size)
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
