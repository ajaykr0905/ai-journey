"""Canonical message and ChatML exports for validated SFT partitions."""

from dataclasses import dataclass
from hashlib import sha256

from .sft_dataset_format import format_dolly_record
from .sft_dataset_identity import canonical_json_bytes
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_split import SFTDatasetSplit
from .sft_dataset_template import render_chatml


@dataclass(frozen=True)
class SFTDatasetArtifacts:
    """The two published payloads: message rows and ChatML rows with loss spans.

    Each must be nonempty, newline-terminated, NUL-free and at most 64 MiB.
    """

    dataset_jsonl: bytes
    chatml_jsonl: bytes

    def __post_init__(self) -> None:
        for name, payload in (
            ("dataset_jsonl", self.dataset_jsonl),
            ("chatml_jsonl", self.chatml_jsonl),
        ):
            if (
                not isinstance(payload, bytes)
                or not payload
                or not payload.endswith(b"\n")
                or b"\x00" in payload
                or len(payload) > 64 * 1024 * 1024
            ):
                raise SFTDatasetError(f"{name} is not bounded newline-terminated bytes")

    def descriptors(self) -> dict[str, dict[str, int | str]]:
        """File name to byte count and SHA-256: the inventory the manifest seals."""
        return {
            name: {
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
            for name, payload in (
                ("dataset.jsonl", self.dataset_jsonl),
                ("chatml.jsonl", self.chatml_jsonl),
            )
        }


def export_split(split: SFTDatasetSplit) -> SFTDatasetArtifacts:
    """Render train then validation records into both JSONL artifacts.

    Rows are canonical JSON tagged with their split, so the bytes depend only on
    the split and can be rebuilt exactly on replay.
    """
    if not isinstance(split, SFTDatasetSplit):
        raise TypeError("split must be SFTDatasetSplit")
    dataset_rows = []
    chatml_rows = []
    for partition, records in (
        ("train", split.train),
        ("validation", split.validation),
    ):
        for record in records:
            example = format_dolly_record(record)
            rendered = render_chatml(example)
            dataset_rows.append(
                canonical_json_bytes({**example.as_record(), "split": partition})
            )
            chatml_rows.append(
                canonical_json_bytes(
                    {
                        "assistant_spans": [
                            list(span) for span in rendered.assistant_spans
                        ],
                        "example_id": example.example_id,
                        "split": partition,
                        "text": rendered.text,
                    }
                )
            )
    return SFTDatasetArtifacts(b"".join(dataset_rows), b"".join(chatml_rows))
