"""Convert Dolly records into strict two-turn chat examples."""

from dataclasses import dataclass
import re

from .sft_dataset_identity import record_id, record_sha256
from .sft_dataset_record import DollyRecord, SFTDatasetError, normalized_text


_EXAMPLE_ID = re.compile(r"dolly-[0-9a-f]{24}")


@dataclass(frozen=True)
class ChatTurn:
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
