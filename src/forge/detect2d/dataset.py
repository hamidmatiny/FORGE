"""Datasets for detect2d training and inference."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from forge.detect2d.model import CLASS_NAMES


class SyntheticDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Randomly generated images + boxes, for CI training-loop smoke tests.

    Not meant to teach the model anything real — it exists to prove the
    LightningModule training step runs correctly on CPU with a valid
    (image, target) batch shape, without needing any real dataset.
    """

    def __init__(
        self,
        num_samples: int = 8,
        image_size: int = 320,
        num_classes: int = len(CLASS_NAMES),
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torch.rand(3, self.image_size, self.image_size)

        num_boxes = self._rng.randint(1, 3)
        boxes = []
        labels = []
        for _ in range(num_boxes):
            x1 = self._rng.uniform(0, self.image_size - 20)
            y1 = self._rng.uniform(0, self.image_size - 20)
            w = self._rng.uniform(10, min(60, self.image_size - x1))
            h = self._rng.uniform(10, min(60, self.image_size - y1))
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(self._rng.randint(1, self.num_classes - 1))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate (image, target) pairs into the list-based batch torchvision detection expects."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def load_image_tensor(path: str, image_size: int = 320) -> torch.Tensor:
    """Load an image file and resize to a fixed square for inference.

    Kept minimal and dependency-light (PIL only, already a torchvision dep) —
    a real training pipeline would use proper letterboxing/augmentation.
    """
    from PIL import Image
    from torchvision.transforms import functional as F

    img: Any = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    tensor: torch.Tensor = F.to_tensor(img)
    return tensor
