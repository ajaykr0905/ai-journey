"""Tensor collation for immutable assistant-only SFT windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_windows import FineTuneWindow


@dataclass(frozen=True)
class FineTuneBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("inputs", "targets", "target_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise TypeError(f"{name} must be a CPU tensor")
        if (
            self.inputs.ndim != 2
            or self.inputs.shape != self.targets.shape
            or self.inputs.shape != self.target_mask.shape
            or self.inputs.dtype != torch.long
            or self.targets.dtype != torch.long
            or self.target_mask.dtype != torch.bool
            or not bool(self.target_mask.any())
        ):
            raise SFTFineTuneError("batch tensors must be aligned assistant targets")


def collate_windows(windows: tuple[FineTuneWindow, ...]) -> FineTuneBatch:
    if not windows or any(not isinstance(row, FineTuneWindow) for row in windows):
        raise SFTFineTuneError("collation requires immutable fine-tuning windows")
    width = len(windows[0].inputs)
    if any(len(row.inputs) != width for row in windows):
        raise SFTFineTuneError("fine-tuning windows must share one width")
    return FineTuneBatch(
        torch.tensor([row.inputs for row in windows], dtype=torch.long),
        torch.tensor([row.targets for row in windows], dtype=torch.long),
        torch.tensor([row.target_mask for row in windows], dtype=torch.bool),
    )
