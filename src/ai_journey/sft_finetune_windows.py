"""Assistant-only byte targets derived from canonical ChatML character spans."""

from __future__ import annotations

from dataclasses import dataclass

from .sft_finetune_codec import UTF8ByteCodec
from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_dataset import FineTuneExample


@dataclass(frozen=True)
class FineTuneWindow:
    example_id: str
    offset: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    target_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        width = len(self.inputs)
        if (
            not self.example_id
            or type(self.offset) is not int
            or self.offset < 0
            or width == 0
            or len(self.targets) != width
            or len(self.target_mask) != width
            or not any(self.target_mask)
        ):
            raise SFTFineTuneError("window must contain aligned assistant targets")


def _byte_spans(example: FineTuneExample) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            len(example.text[:start].encode("utf-8")),
            len(example.text[:end].encode("utf-8")),
        )
        for start, end in example.assistant_spans
    )


def example_windows(
    example: FineTuneExample, *, block_size: int, codec: UTF8ByteCodec | None = None
) -> tuple[FineTuneWindow, ...]:
    """Return non-crossing windows whose selected targets are assistant bytes."""
    if not isinstance(example, FineTuneExample):
        raise TypeError("example must be FineTuneExample")
    if type(block_size) is not int or block_size <= 0:
        raise SFTFineTuneError("block_size must be a positive integer")
    codec = codec or UTF8ByteCodec()
    tokens = codec.encode(example.text)
    spans = _byte_spans(example)
    width = min(block_size, len(tokens) - 1)
    if width <= 0:
        raise SFTFineTuneError("example requires at least two encoded bytes")
    max_offset = len(tokens) - width - 1
    offsets = set(range(0, max_offset + 1, width))
    offsets.add(max_offset)
    windows = []
    for offset in sorted(offsets):
        inputs = tokens[offset : offset + width]
        targets = tokens[offset + 1 : offset + width + 1]
        if len(inputs) != len(targets):
            continue
        mask = tuple(
            any(start <= target_offset < end for start, end in spans)
            for target_offset in range(offset + 1, offset + 1 + len(targets))
        )
        if any(mask):
            windows.append(
                FineTuneWindow(example.example_id, offset, inputs, targets, mask)
            )
    if not windows:
        raise SFTFineTuneError("example has no trainable assistant window")
    return tuple(windows)
