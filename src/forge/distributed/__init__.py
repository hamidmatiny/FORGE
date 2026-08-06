"""Ray-backed local distributed execution — no cloud cluster required."""

from forge.distributed.ray_utils import run_distributed_map

__all__ = ["run_distributed_map"]
