"""Strict source records for public supervised fine-tuning datasets."""

from dataclasses import dataclass
import re
import unicodedata


class SFTDatasetError(ValueError):
    """An SFT dataset input or artifact violates its public contract."""


_CATEGORIES = frozenset(
    {
        "brainstorming",
        "classification",
        "closed_qa",
        "creative_writing",
        "general_qa",
        "information_extraction",
        "open_qa",
        "summarization",
    }
)
# Every C0 control except tab and newline; one regex scan checks a whole field.
_UNSUPPORTED_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")


def normalized_text(value: str, field: str, *, allow_empty: bool = False) -> str:
    """Return bounded NFC text with canonical newlines and no control bytes."""
    if not isinstance(value, str):
        raise SFTDatasetError(f"{field} must be text")
    text = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    if not text and not allow_empty:
        raise SFTDatasetError(f"{field} must not be empty")
    if len(text.encode("utf-8")) > 64 * 1024:
        raise SFTDatasetError(f"{field} exceeds the UTF-8 byte limit")
    if _UNSUPPORTED_CONTROL.search(text):
        raise SFTDatasetError(f"{field} contains an unsupported control character")
    return text


@dataclass(frozen=True)
class DollyRecord:
    """One normalized Databricks Dolly instruction record."""

    instruction: str
    context: str
    response: str
    category: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instruction", normalized_text(self.instruction, "instruction")
        )
        object.__setattr__(
            self,
            "context",
            normalized_text(self.context, "context", allow_empty=True),
        )
        object.__setattr__(self, "response", normalized_text(self.response, "response"))
        if not isinstance(self.category, str) or self.category not in _CATEGORIES:
            raise SFTDatasetError("category is not a supported Dolly task category")

    @classmethod
    def from_mapping(cls, value: object) -> "DollyRecord":
        if not isinstance(value, dict) or set(value) != {
            "instruction",
            "context",
            "response",
            "category",
        }:
            raise SFTDatasetError("Dolly records require exactly four source fields")
        return cls(**value)
