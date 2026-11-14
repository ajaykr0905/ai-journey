"""Convert Dolly records into strict two-turn chat examples."""

from dataclasses import dataclass
import re

from .sft_dataset_identity import record_id, record_sha256
from .sft_dataset_record import DollyRecord, SFTDatasetError, normalized_text


_EXAMPLE_ID = re.compile(r"dolly-[0-9a-f]{24}")


@dataclass(frozen=True)
class ChatTurn:
    """One user or assistant message, renormalized like source text on creation."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise SFTDatasetError(
                "SFT chat turns support only user and assistant roles"
            )
        object.__setattr__(
            self, "content", normalized_text(self.content, "chat content")
        )


@dataclass(frozen=True)
class ChatExample:
    """A strict user-then-assistant pair tied to its source record's identity.

    example_id and source_sha256 must have the Dolly id and digest formats, so
    every exported row traces back to one exact source record.
    """

    example_id: str
    source_sha256: str
    category: str
    messages: tuple[ChatTurn, ChatTurn]

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not _EXAMPLE_ID.fullmatch(
            self.example_id
        ):
            raise SFTDatasetError("example_id must be a stable Dolly identifier")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_sha256
            )
        ):
            raise SFTDatasetError("source_sha256 must be lowercase hexadecimal")
        if not isinstance(self.category, str) or not self.category:
            raise SFTDatasetError("chat category must be nonempty text")
        if (
            not isinstance(self.messages, tuple)
            or len(self.messages) != 2
            or any(not isinstance(message, ChatTurn) for message in self.messages)
            or tuple(message.role for message in self.messages) != ("user", "assistant")
        ):
            raise SFTDatasetError(
                "SFT examples require one user turn then one assistant turn"
            )

    def as_record(self) -> dict:
        """Message-format row for dataset.jsonl; export adds the split field."""
        return {
            "category": self.category,
            "example_id": self.example_id,
            "messages": [
                {"content": message.content, "role": message.role}
                for message in self.messages
            ],
            "source_sha256": self.source_sha256,
        }


def format_dolly_record(record: DollyRecord) -> ChatExample:
    """Turn a record into one user turn and one assistant turn.

    Nonempty context follows the instruction under a "Context:" heading, and the
    ids come from the record's content digest.
    """
    if not isinstance(record, DollyRecord):
        raise TypeError("record must be DollyRecord")
    prompt = record.instruction
    if record.context:
        prompt = f"{prompt}\n\nContext:\n{record.context}"
    return ChatExample(
        example_id=record_id(record),
        source_sha256=record_sha256(record),
        category=record.category,
        messages=(ChatTurn("user", prompt), ChatTurn("assistant", record.response)),
    )
