"""Ray-backed distributed execution — local multi-process, no cluster/cloud required.

Cost-safety: this never provisions a real Ray cluster or touches cloud
resources. ``ray.init()`` with no address starts a purely local instance
using this machine's own CPU cores — the same "no real cloud/GPU spend"
policy every other phase follows. A real Ray cluster (EC2/EKS-backed) is
what a production deployment would point this at instead, by passing a
cluster address to ``ray.init()`` — deliberately not built here, since
provisioning and paying for one isn't something this project does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import ray

T = TypeVar("T")
R = TypeVar("R")


def run_distributed_map(
    fn: Callable[[T], R],
    items: list[T],
    distributed: bool,
    num_cpus: int | None = None,
) -> list[R]:
    """Apply `fn` to every item in `items`, in parallel via local Ray if `distributed`.

    Args:
        fn: A function of one argument. When `distributed=True`, this gets
            wrapped with `ray.remote` — it must be picklable (cloudpickle),
            which ordinary functions and closures over picklable objects
            (including PyTorch models) satisfy.
        items: The work items to map over.
        distributed: If False, runs a plain sequential Python loop —
            identical results either way, just a different execution
            strategy. If True, distributes across local CPU cores via Ray.
        num_cpus: Cap the number of local Ray worker processes. `None`
            lets Ray use all available cores.

    Returns:
        Results in the same order as `items`.
    """
    if not distributed or not items:
        return [fn(item) for item in items]

    was_already_initialized = ray.is_initialized()
    if not was_already_initialized:
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True, log_to_driver=False)

    try:
        remote_fn = ray.remote(fn)
        futures = [remote_fn.remote(item) for item in items]
        results: list[R] = ray.get(futures)
        return results
    finally:
        if not was_already_initialized:
            ray.shutdown()
