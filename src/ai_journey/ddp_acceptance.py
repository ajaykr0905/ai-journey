"""Executable acceptance oracles for real multi-process training and recovery."""

from pathlib import Path

import torch

from .ddp_checkpoint import bounded_checkpoint_load
from .ddp_launch import launch_training


def exact_state(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exact_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact_state(a, b) for a, b in zip(left, right)
        )
    return left == right


def certify_recovery(
    *, source: Path, spec: Path, directory: Path, world_size=2, split_steps=2
) -> dict:
    """Compare an uninterrupted run to a stopped run resumed in fresh processes."""
    continuous_path = directory / "continuous.pt"
    resumed_path = directory / "resumed.pt"
    continuous = launch_training(
        source=source,
        spec=spec,
        checkpoint=continuous_path,
        output_dir=directory / "continuous",
        world_size=world_size,
        steps=split_steps * 2,
    )
    first = launch_training(
        source=source,
        spec=spec,
        checkpoint=resumed_path,
        output_dir=directory / "first",
        world_size=world_size,
        steps=split_steps,
    )
    resumed = launch_training(
        source=source,
        spec=spec,
        checkpoint=resumed_path,
        output_dir=directory / "resumed",
        world_size=world_size,
        steps=split_steps,
        resume=True,
    )
    expected = bounded_checkpoint_load(continuous_path, 64 * 1024 * 1024)
    actual = bounded_checkpoint_load(resumed_path, 64 * 1024 * 1024)
    verified = {
        name: exact_state(expected[name], actual[name])
        for name in ("model", "optimizer", "rank_rngs", "step")
    }
    if not all(verified.values()):
        raise RuntimeError(
            "distributed recovery does not reproduce uninterrupted state"
        )
    if {rank["pid"] for rank in first} & {rank["pid"] for rank in resumed}:
        raise RuntimeError("restart evidence requires fresh worker processes")
    return {
        "world_size": world_size,
        "completed_step": split_steps * 2,
        "exact_state": verified,
        "model_sha256": continuous[0]["model_sha256"],
        "initial_worker_pids": [rank["pid"] for rank in first],
        "resumed_worker_pids": [rank["pid"] for rank in resumed],
    }
