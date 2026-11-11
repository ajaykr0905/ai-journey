"""Deterministic, evidence-bearing sampling of assistant-only SFT windows."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256

from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_dataset import FineTuneExample
from .sft_finetune_windows import FineTuneWindow, example_windows


@dataclass(frozen=True)
class FineTuneSchedule:
    batches: tuple[tuple[FineTuneWindow, ...], ...]
    sha256: str

    def __post_init__(self) -> None:
        if not self.batches or not self.sha256:
            raise SFTFineTuneError("schedule requires batches and a digest")
        size = len(self.batches[0])
        if size == 0 or any(len(batch) != size for batch in self.batches):
            raise SFTFineTuneError("schedule batches must have one fixed size")


def build_schedule(
    examples: tuple[FineTuneExample, ...],
    *,
    block_size: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> FineTuneSchedule:
    if not examples or any(row.split != "train" for row in examples):
        raise SFTFineTuneError("schedule requires training examples")
    if any(type(value) is not int or value <= 0 for value in (steps, batch_size)):
        raise SFTFineTuneError("steps and batch_size must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise SFTFineTuneError("seed is outside the deterministic range")
    candidates = tuple(
        window
        for example in examples
        for window in example_windows(example, block_size=block_size)
    )
    generator = random.Random(seed)
    batches = tuple(
        tuple(
            candidates[generator.randrange(len(candidates))] for _ in range(batch_size)
        )
        for _ in range(steps)
    )
    evidence = [[[row.example_id, row.offset] for row in batch] for batch in batches]
    digest = sha256(
        json.dumps(evidence, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FineTuneSchedule(batches, digest)
