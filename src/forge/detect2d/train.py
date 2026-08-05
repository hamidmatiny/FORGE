"""CPU training loop for the 2D detector — smoke-tests the pipeline mechanics.

No real GPU spend and no dataset download: trains on
``SyntheticDetectionDataset`` for a small, bounded number of steps. This
proves the LightningModule, optimizer, and checkpoint save/load path all
work correctly — it does not produce an accurate detector. Training against
real labeled data (from a future ``forge label`` / human-reviewed set) is
tracked in KNOWN_GAPS.md.
"""

from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader

from forge.detect2d.dataset import SyntheticDetectionDataset, detection_collate
from forge.detect2d.model import CLASS_NAMES, Detector2DModule


def train_detector(
    output_checkpoint: Path,
    max_steps: int = 10,
    num_classes: int = len(CLASS_NAMES),
    num_samples: int = 8,
    batch_size: int = 2,
    lr: float = 1e-4,
    seed: int = 0,
) -> float:
    """Run a short CPU training loop and save the resulting checkpoint.

    Returns:
        The final training-step loss (for logging/smoke assertions).
    """
    torch.manual_seed(seed)

    dataset = SyntheticDetectionDataset(num_samples=num_samples, num_classes=num_classes, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate,
    )

    module = Detector2DModule(num_classes=num_classes, lr=lr)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=max_steps,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": module.model.state_dict(),
            "num_classes": num_classes,
        },
        output_checkpoint,
    )

    final_loss = trainer.callback_metrics.get("train_loss")
    return float(final_loss) if final_loss is not None else float("nan")
