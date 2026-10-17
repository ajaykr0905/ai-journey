"""Validated, public-safe conversation data for bounded chat SFT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re


class SFTDataError(ValueError):
    """Conversation data violates the supervised fine-tuning contract."""


_CONVERSATION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self):
        if self.role not in _ROLES:
            raise SFTDataError(f"unsupported chat role: {self.role!r}")
        if (
            not isinstance(self.content, str)
            or not self.content.strip()
            or len(self.content) > 4096
            or "\x00" in self.content
            or "\r" in self.content
        ):
            raise SFTDataError(
                "message content must be nonempty normalized text up to 4096 characters"
            )


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    messages: tuple[ChatMessage, ...]

    def __post_init__(self):
        if (
            not isinstance(self.conversation_id, str)
            or _CONVERSATION_ID.fullmatch(self.conversation_id) is None
        ):
            raise SFTDataError(
                "conversation_id must be 1-64 lowercase identifier characters"
            )
        if (
            not isinstance(self.messages, tuple)
            or not self.messages
            or any(not isinstance(message, ChatMessage) for message in self.messages)
        ):
            raise SFTDataError("messages must be a nonempty tuple of ChatMessage values")
        system_count = sum(message.role == "system" for message in self.messages)
        if system_count > 1 or (
            system_count == 1 and self.messages[0].role != "system"
        ):
            raise SFTDataError("at most one system message may appear, and only first")
        dialogue = self.messages[1:] if system_count else self.messages
        if not dialogue or dialogue[0].role != "user":
            raise SFTDataError("dialogue must start with a user message")
        expected = ("user", "assistant")
        for index, message in enumerate(dialogue):
            if message.role != expected[index % 2]:
                raise SFTDataError("dialogue must alternate user and assistant messages")
        if dialogue[-1].role != "assistant":
            raise SFTDataError("conversation must end with an assistant response")
