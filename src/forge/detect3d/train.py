"""CPU training loop for the 3D detector — smoke-tests the pipeline mechanics.

Same honesty as detect2d/train.py: trains on synthetic in-memory point
clouds for a bounded number of steps to prove the LightningModule,
optimizer, and checkpoint path all work. Does not produce an accurate
detector — see DECISIONS.md ADR-014 and KNOWN_GAPS.md.
"""

from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader

from forge.detect3d.dataset import SyntheticPointCloudDataset, point3d_collate
from forge.detect3d.model import CLASS_NAMES, NUM_QUERIES, Detector3DModule


def train_detector(
    output_checkpoint: Path,
    max_steps: int = 10,
    num_classes: int = len(CLASS_NAMES),
    num_queries: int = NUM_QUERIES,
    num_samples: int = 8,
    batch_size: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
) -> float:
    """Run a short CPU training loop and save the resulting checkpoint.

    Returns:
        The final training-step loss (for logging/smoke assertions).
    """
    torch.manual_seed(seed)

    dataset = SyntheticPointCloudDataset(
        num_samples=num_samples, num_queries=num_queries, num_classes=num_classes, seed=seed
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=point3d_collate)

    module = Detector3DModule(num_classes=num_classes, num_queries=num_queries, lr=lr)

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
            "model_state_dict": module.state_dict(),
            "num_classes": num_classes,
            "num_queries": num_queries,
        },
        output_checkpoint,
    )

    final_loss = trainer.callback_metrics.get("train_loss")
    return float(final_loss) if final_loss is not None else float("nan")
