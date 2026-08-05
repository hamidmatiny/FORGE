"""Datasets for detect3d training."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from forge.detect3d.model import BOX_DIM, CLASS_NAMES, NUM_QUERIES, POINT_FEATURES


class SyntheticPointCloudDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Randomly generated point clouds + exactly NUM_QUERIES boxes each.

    Like detect2d's SyntheticDetectionDataset, this exists to smoke-test the
    training loop mechanics on CPU with no real dataset, not to teach the
    model anything real. Every sample has exactly ``num_queries`` boxes (see
    DECISIONS.md ADR-014) so each query slot has a direct training target.
    """

    def __init__(
        self,
        num_samples: int = 8,
        num_points: int = 200,
        num_queries: int = NUM_QUERIES,
        num_classes: int = len(CLASS_NAMES),
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.rand(self.num_points, POINT_FEATURES) * torch.tensor(
            [80.0, 80.0, 4.0, 1.0]
        ) - torch.tensor([40.0, 40.0, 2.0, 0.0])

        classes = torch.tensor(
            [self._rng.randint(0, self.num_classes - 1) for _ in range(self.num_queries)],
            dtype=torch.int64,
        )
        boxes = torch.zeros(self.num_queries, BOX_DIM)
        for i in range(self.num_queries):
            boxes[i, 0:3] = torch.tensor(
                [self._rng.uniform(-40, 40), self._rng.uniform(-40, 40), self._rng.uniform(-1, 2)]
            )
            boxes[i, 3:6] = torch.tensor(
                [self._rng.uniform(1.5, 2.5), self._rng.uniform(1.4, 1.8), self._rng.uniform(3, 5)]
            )
            boxes[i, 6] = self._rng.uniform(-3.14159, 3.14159)

        return points, classes, boxes


def point3d_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate (points, classes, boxes) triples: point clouds stay a list (variable N)."""
    point_clouds = [item[0] for item in batch]
    classes = torch.stack([item[1] for item in batch])
    boxes = torch.stack([item[2] for item in batch])
    return point_clouds, classes, boxes
