"""Validated, public-safe conversation data for bounded chat SFT experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class SFTDataError(ValueError):
    """Conversation data violates the supervised fine-tuning contract."""


_CONVERSATION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_TOOL_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}")
_CALL_ID = re.compile(r"call_[a-zA-Z0-9_-]{1,59}")


def _strict_json_object(value: str) -> dict:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise SFTDataError(f"duplicate tool argument: {key}")
            result[key] = item
        return result

    def nonfinite(value):
        raise SFTDataError(f"nonfinite tool argument: {value}")

    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise SFTDataError("tool arguments must be bounded UTF-8 JSON")
    try:
        decoded = json.loads(value, object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SFTDataError(f"invalid tool arguments: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SFTDataError("tool arguments must be a JSON object")
    return decoded


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str

    def __post_init__(self):
        if (
            not isinstance(self.call_id, str)
            or _CALL_ID.fullmatch(self.call_id) is None
        ):
            raise SFTDataError("tool call_id must use the call_ identifier format")
        if not isinstance(self.name, str) or _TOOL_NAME.fullmatch(self.name) is None:
            raise SFTDataError("tool name must be a bounded dotted identifier")
        _strict_json_object(self.arguments)

    @property
    def canonical_arguments(self) -> str:
        return json.dumps(
            _strict_json_object(self.arguments),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_call: ToolCall | None = None
    tool_call_id: str | None = None

    def __post_init__(self):
        if self.role not in _ROLES:
            raise SFTDataError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str):
            raise SFTDataError("message content must be text")
        permits_empty = self.role == "assistant" and self.tool_call is not None
        if (
            (not self.content.strip() and not permits_empty)
            or len(self.content) > 4096
            or "\x00" in self.content
            or "\r" in self.content
        ):
            raise SFTDataError(
                "message content must be nonempty normalized text up to 4096 characters"
            )
        if self.role == "assistant":
            if self.tool_call is not None and not isinstance(self.tool_call, ToolCall):
                raise SFTDataError("assistant tool_call must be a ToolCall")
            if self.tool_call_id is not None:
                raise SFTDataError("assistant messages cannot be tool results")
        elif self.role == "tool":
            if self.tool_call is not None:
                raise SFTDataError("tool results cannot request another tool")
            if (
                not isinstance(self.tool_call_id, str)
                or _CALL_ID.fullmatch(self.tool_call_id) is None
            ):
                raise SFTDataError("tool results require the originating call_id")
        elif self.tool_call is not None or self.tool_call_id is not None:
            raise SFTDataError(
                "tool metadata is allowed only on assistant or tool messages"
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
            raise SFTDataError(
                "messages must be a nonempty tuple of ChatMessage values"
            )
        system_count = sum(message.role == "system" for message in self.messages)
        if system_count > 1 or (
            system_count == 1 and self.messages[0].role != "system"
        ):
            raise SFTDataError("at most one system message may appear, and only first")
        dialogue = self.messages[1:] if system_count else self.messages
        if not dialogue or dialogue[0].role != "user":
            raise SFTDataError("dialogue must start with a user message")
        expected = "user"
        pending_call = None
        seen_calls = set()
        for message in dialogue:
            if message.role != expected:
                raise SFTDataError(f"expected {expected} message, got {message.role}")
            if message.role == "user":
                expected = "assistant"
            elif message.role == "assistant" and message.tool_call is not None:
                pending_call = message.tool_call.call_id
                if pending_call in seen_calls:
                    raise SFTDataError(
                        "tool call ids must be unique within a conversation"
                    )
                seen_calls.add(pending_call)
                expected = "tool"
            elif message.role == "tool":
                if message.tool_call_id != pending_call:
                    raise SFTDataError("tool result call_id does not match the request")
                pending_call = None
                expected = "assistant"
            else:
                expected = "user"
        if dialogue[-1].role != "assistant" or dialogue[-1].tool_call is not None:
            raise SFTDataError("conversation must end with an assistant response")
