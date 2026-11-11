"""Deterministic held-out coverage for public SFT comparisons."""

from __future__ import annotations

import json
from hashlib import sha256

from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_dataset import FineTuneExample
from .sft_finetune_windows import FineTuneWindow, example_windows


def validation_windows(
    examples: tuple[FineTuneExample, ...], *, block_size: int, example_limit: int
) -> tuple[FineTuneWindow, ...]:
    if not examples or any(row.split != "validation" for row in examples):
        raise SFTFineTuneError("held-out coverage requires validation examples")
    if type(example_limit) is not int or not 1 <= example_limit <= len(examples):
        raise SFTFineTuneError("example_limit must fit the validation partition")
    return tuple(
        window
        for example in examples[:example_limit]
        for window in example_windows(example, block_size=block_size)
    )


def validation_evidence(windows: tuple[FineTuneWindow, ...]) -> dict:
    if not windows:
        raise SFTFineTuneError("held-out evidence requires windows")
    identities = [[row.example_id, row.offset] for row in windows]
    return {
        "examples": len({row.example_id for row in windows}),
        "windows": len(windows),
        "target_tokens": sum(sum(row.target_mask) for row in windows),
        "sha256": sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest(),
    }
