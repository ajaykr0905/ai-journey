"""Transactional deterministic optimization over a fixed SFT schedule."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math

import torch

from .sft_finetune_batch import collate_windows
from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_optimizer import build_finetune_optimizer
from .sft_finetune_schedule import FineTuneSchedule
from .transformer_lab import DecoderLanguageModel, model_fingerprint


def train_finetune_schedule(
    model: DecoderLanguageModel,
    schedule: FineTuneSchedule,
    config: SFTFineTuneConfig,
    *,
    optimizer: torch.optim.AdamW | None = None,
) -> tuple[torch.optim.AdamW, dict]:
    if len(schedule.batches) != config.steps:
        raise SFTFineTuneError("schedule steps do not match the fine-tuning config")
    optimizer = optimizer or build_finetune_optimizer(model, config)
    losses = []
    gradient_norms = []
    clipped_steps = 0
    model.train()
    for windows in schedule.batches:
        batch = collate_windows(windows)
        model_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        try:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(batch.inputs, batch.targets, loss_mask=batch.target_mask)
            if loss is None or not bool(torch.isfinite(loss)):
                raise SFTFineTuneError("training loss must remain finite")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            )
            if not math.isfinite(norm):
                raise SFTFineTuneError("gradient norm must remain finite")
            clipped_steps += norm > config.gradient_clip
            optimizer.step()
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            ):
                raise SFTFineTuneError("optimizer produced nonfinite model state")
            losses.append(float(loss.detach()))
            gradient_norms.append(norm)
        except Exception:
            model.load_state_dict(model_before)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            raise
    return optimizer, {
        "losses": losses,
        "gradient_norms": gradient_norms,
        "clipped_steps": clipped_steps,
        "optimizer_steps": len(losses),
        "input_tokens": config.steps * config.batch_size * config.block_size,
        "supervised_tokens": sum(
            sum(sum(row.target_mask) for row in batch) for batch in schedule.batches
        ),
        "final_model_sha256": model_fingerprint(model),
    }


def train_finetune_segment(
    model: DecoderLanguageModel,
    batches: tuple,
    config: SFTFineTuneConfig,
    *,
    optimizer: torch.optim.AdamW | None = None,
) -> tuple[torch.optim.AdamW, dict]:
    """Train a contiguous schedule segment for exact checkpoint continuation."""
    if not batches or len(batches) > config.steps:
        raise SFTFineTuneError("segment must contain one to config.steps batches")
    segment = FineTuneSchedule(
        batches,
        hashlib.sha256(
            repr(
                [[(row.example_id, row.offset) for row in batch] for batch in batches]
            ).encode()
        ).hexdigest(),
    )
    segment_config = dataclasses.replace(
        config,
        steps=len(batches),
        checkpoint_interval=min(config.checkpoint_interval, len(batches)),
    )
    return train_finetune_schedule(model, segment, segment_config, optimizer=optimizer)
