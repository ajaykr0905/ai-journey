"""Strict loading of the sealed ChatML dataset produced by Day 67."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .sft_finetune_contract import SFTFineTuneError

_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FineTuneExample:
    example_id: str
    split: str
    text: str
    assistant_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise SFTFineTuneError("example_id must be nonempty text")
        if self.split not in {"train", "validation"}:
            raise SFTFineTuneError("split must be train or validation")
        if not isinstance(self.text, str) or not self.text:
            raise SFTFineTuneError("example text must be nonempty")
        previous = 0
        for span in self.assistant_spans:
            if (
                not isinstance(span, tuple)
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1] <= len(self.text)
            ):
                raise SFTFineTuneError("assistant spans must be ordered text intervals")
            previous = span[1]
        if not self.assistant_spans:
            raise SFTFineTuneError("each example requires assistant supervision")


@dataclass(frozen=True)
class FineTuneDataset:
    examples: tuple[FineTuneExample, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not self.examples or len({row.example_id for row in self.examples}) != len(
            self.examples
        ):
            raise SFTFineTuneError("dataset requires unique examples")
        if {row.split for row in self.examples} != {"train", "validation"}:
            raise SFTFineTuneError("dataset requires train and validation examples")
        if _DIGEST.fullmatch(self.sha256) is None:
            raise SFTFineTuneError("dataset sha256 is invalid")

    @property
    def train(self) -> tuple[FineTuneExample, ...]:
        return tuple(row for row in self.examples if row.split == "train")

    @property
    def validation(self) -> tuple[FineTuneExample, ...]:
        return tuple(row for row in self.examples if row.split == "validation")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SFTFineTuneError(f"duplicate dataset field: {key}")
        value[key] = item
    return value


def load_chatml_dataset(
    path: Path, *, max_bytes: int = 64 * 1024 * 1024
) -> FineTuneDataset:
    """Load bounded regular JSONL bytes and bind every parsed row to their digest."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if path.is_symlink() or not path.is_file():
        raise SFTFineTuneError("ChatML dataset must be a regular non-symlink file")
    payload = path.read_bytes()
    if not 1 <= len(payload) <= max_bytes or not payload.endswith(b"\n"):
        raise SFTFineTuneError(
            "ChatML dataset must be bounded newline-terminated bytes"
        )
    examples = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    SFTFineTuneError(f"nonfinite dataset value: {token}")
                ),
            )
            if not isinstance(value, dict) or set(value) != {
                "assistant_spans",
                "example_id",
                "split",
                "text",
            }:
                raise SFTFineTuneError("dataset row has unexpected fields")
            spans = value["assistant_spans"]
            if not isinstance(spans, list):
                raise SFTFineTuneError("assistant_spans must be a list")
            examples.append(
                FineTuneExample(
                    value["example_id"],
                    value["split"],
                    value["text"],
                    tuple(
                        tuple(span) if isinstance(span, list) else span
                        for span in spans
                    ),
                )
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, SFTFineTuneError) as exc:
            raise SFTFineTuneError(f"invalid ChatML row {line_number}: {exc}") from exc
    return FineTuneDataset(tuple(examples), sha256(payload).hexdigest())
