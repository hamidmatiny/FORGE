"""Tests for forge.distributed.run_distributed_map.

The sequential (`distributed=False`) path and the empty-items shortcut are
tested for real — no mocking needed, since neither touches Ray at all.

The actual Ray (`distributed=True`, non-empty items) path is tested with
Ray's API mocked rather than exercised for real: this development sandbox
cannot run Ray's multi-process runtime at all (its C++ core's plasma
object store hangs/crashes here regardless of configuration — see
DECISIONS.md for what was tried). Mocking verifies *this module's* logic
is correct (init/remote/get/shutdown called with the right arguments in
the right order) without depending on Ray's own internals actually
working in this environment — the same boundary-mocking approach used for
AWS in tests/test_lambda_ingest_trigger.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ray")

from forge.distributed import run_distributed_map  # noqa: E402


def test_sequential_matches_plain_map() -> None:
    def square(x: int) -> int:
        return x * x

    result = run_distributed_map(square, [1, 2, 3, 4], distributed=False)
    assert result == [1, 4, 9, 16]


def test_sequential_preserves_order() -> None:
    def negate(x: int) -> int:
        return -x

    result = run_distributed_map(negate, [5, 1, 3], distributed=False)
    assert result == [-5, -1, -3]


def test_empty_items_short_circuits_without_touching_ray() -> None:
    """Empty input never calls ray at all, even with distributed=True."""

    def boom(x: int) -> int:
        raise AssertionError("should never be called")

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        result = run_distributed_map(boom, [], distributed=True)
        assert result == []
        mock_ray.init.assert_not_called()
        mock_ray.get.assert_not_called()


def test_distributed_initializes_ray_when_not_already_running() -> None:
    def square(x: int) -> int:
        return x * x

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_remote_fn = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_remote_fn.remote.side_effect = lambda item: f"future-{item}"
        mock_ray.get.return_value = [1, 4, 9]

        result = run_distributed_map(square, [1, 2, 3], distributed=True, num_cpus=2)

        mock_ray.init.assert_called_once_with(
            num_cpus=2, ignore_reinit_error=True, log_to_driver=False
        )
        mock_ray.shutdown.assert_called_once()
        assert result == [1, 4, 9]


def test_distributed_skips_init_and_shutdown_when_already_running() -> None:
    """If Ray is already initialized by the caller, don't touch its lifecycle."""

    def square(x: int) -> int:
        return x * x

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = True
        mock_remote_fn = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_ray.get.return_value = [1, 4]

        run_distributed_map(square, [1, 2], distributed=True)

        mock_ray.init.assert_not_called()
        mock_ray.shutdown.assert_not_called()


def test_distributed_wraps_fn_once_and_calls_remote_per_item() -> None:
    def square(x: int) -> int:
        return x * x

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_remote_fn = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_ray.get.return_value = [1, 4, 9, 16]

        run_distributed_map(square, [1, 2, 3, 4], distributed=True)

        mock_ray.remote.assert_called_once_with(square)
        assert mock_remote_fn.remote.call_count == 4
        mock_remote_fn.remote.assert_any_call(1)
        mock_remote_fn.remote.assert_any_call(4)


def test_distributed_shuts_down_ray_even_if_get_raises() -> None:
    def square(x: int) -> int:
        return x * x

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_ray.remote.return_value = MagicMock()
        mock_ray.get.side_effect = RuntimeError("simulated worker crash")

        with pytest.raises(RuntimeError, match="simulated worker crash"):
            run_distributed_map(square, [1, 2], distributed=True)

        mock_ray.shutdown.assert_called_once()


# --- shared_args: large objects go through ray.put(), not the closure -----


def test_sequential_passes_shared_args_directly() -> None:
    """No Ray involved -- shared_args just get passed through as extra arguments."""

    def add(x: int, offset: int, label: str) -> str:
        return f"{label}:{x + offset}"

    result = run_distributed_map(add, [1, 2, 3], distributed=False, shared_args=(10, "n"))
    assert result == ["n:11", "n:12", "n:13"]


def test_distributed_puts_each_shared_arg_exactly_once() -> None:
    def add(x: int, offset: int) -> int:
        return x + offset

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_ray.put.side_effect = lambda arg: f"ref-{arg}"
        mock_remote_fn = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_ray.get.return_value = [11, 12, 13]

        run_distributed_map(add, [1, 2, 3], distributed=True, shared_args=(10,))

        # put() called once for the shared arg, not once per item.
        mock_ray.put.assert_called_once_with(10)
        # Every item call gets the SAME put()-returned ref, not the raw value.
        mock_remote_fn.remote.assert_any_call(1, "ref-10")
        mock_remote_fn.remote.assert_any_call(2, "ref-10")
        mock_remote_fn.remote.assert_any_call(3, "ref-10")


def test_distributed_with_no_shared_args_never_calls_put() -> None:
    def square(x: int) -> int:
        return x * x

    with patch("forge.distributed.ray_utils.ray") as mock_ray:
        mock_ray.is_initialized.return_value = False
        mock_remote_fn = MagicMock()
        mock_ray.remote.return_value = mock_remote_fn
        mock_ray.get.return_value = [1, 4]

        run_distributed_map(square, [1, 2], distributed=True)

        mock_ray.put.assert_not_called()
