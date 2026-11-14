"""Materialize only one declared byte optimizer window, not the whole corpus."""

import torch

from .ddp_contracts import Topology
from .ddp_data import shard_indices
from .moe_ddp_plan import MAX_SAMPLES, SAMPLER_V2, source_admission


def public_moe_ddp_schedule(plan, source_bytes, *, rank, update_index):
    """Sample IDs one rank consumes for one optimizer update, without data.

    The epoch and offset follow from the update index. V1 truncates the
    original IDs to complete global windows before shuffling, kept so existing
    checkpoints replay; v2 shuffles all IDs. Raises ValueError for a cursor or
    source size outside the declared schedule.
    """
    run = plan.run
    if (
        type(source_bytes) is not int
        or not 1 < source_bytes <= 1024 * 1024
        or type(update_index) is not int
        or not 0 <= update_index < run.optimizer_steps
    ):
        raise ValueError("public MoE data cursor is outside its declared schedule")
    count = (source_bytes - 1) // run.sequence_length
    global_batch = run.world_size * run.micro_batch_size * run.accumulation_steps
    if not global_batch <= count <= MAX_SAMPLES:
        raise ValueError("public MoE schedule exceeds its sample budget")
    updates = count // global_batch
    epoch, offset = divmod(update_index, updates)
    ids = shard_indices(
        # V1 truncated original IDs; retain it for existing checkpoint replay.
        # V2 shuffles all IDs. Window slices consume only complete global groups.
        count if plan.sampler == SAMPLER_V2 else count - count % global_batch,
        Topology(rank, run.world_size),
        seed=run.seed,
        epoch=epoch,
    )
    begin = offset * run.accumulation_steps * run.micro_batch_size
    return {
        "epoch": epoch,
        "update_offset": offset,
        "sample_ids": [
            list(
                ids[
                    begin + index * run.micro_batch_size : begin
                    + (index + 1) * run.micro_batch_size
                ]
            )
            for index in range(run.accumulation_steps)
        ],
    }


def public_moe_ddp_window(plan, source, *, rank, update_index):
    """Materialize only one rank's microbatches for one optimizer update.

    The source is re-admitted against the plan digest first. Each sample is a
    next-byte input/target pair with every position eligible. Returns the
    batches and their schedule.
    """
    source_admission(plan, source)
    schedule = public_moe_ddp_schedule(
        plan, len(source), rank=rank, update_index=update_index
    )
    length = plan.run.sequence_length
    batches = []
    for ids in schedule["sample_ids"]:
        inputs = torch.tensor(
            [
                list(source[sample * length : sample * length + length])
                for sample in ids
            ],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [
                list(source[sample * length + 1 : sample * length + length + 1])
                for sample in ids
            ],
            dtype=torch.long,
        )
        batches.append((inputs, targets, torch.ones_like(inputs, dtype=torch.bool)))
    return tuple(batches), schedule
