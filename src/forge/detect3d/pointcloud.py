"""Loading nuScenes-format lidar point clouds."""

from __future__ import annotations

import numpy as np
import torch

# nuScenes LIDAR_TOP .pcd.bin files are raw float32 arrays with 5 columns
# per point: x, y, z, intensity, ring_index.
POINT_COLUMNS = 5


def load_point_cloud(path: str) -> torch.Tensor:
    """Load a nuScenes-format .pcd.bin file as an (N, 5) float32 tensor."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % POINT_COLUMNS != 0:
        raise ValueError(
            f"{path}: {raw.size} float32 values is not a multiple of "
            f"{POINT_COLUMNS} (x, y, z, intensity, ring)."
        )
    points = raw.reshape(-1, POINT_COLUMNS)
    return torch.from_numpy(points.copy())
