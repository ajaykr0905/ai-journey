"""Validated, public-safe conversation data for bounded chat SFT experiments."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from hashlib import sha256

from .transformer_lab import CharacterCodec


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


@dataclass(frozen=True)
class RenderedConversation:
    conversation_id: str
    text: str
    assistant_spans: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.endswith("\n"):
            raise SFTDataError("rendered conversations must end with a newline")
        previous_end = 0
        for span in self.assistant_spans:
            if (
                not isinstance(span, tuple)
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or not previous_end <= span[0] < span[1] <= len(self.text)
            ):
                raise SFTDataError("assistant spans must be ordered nonempty intervals")
            previous_end = span[1]

    @property
    def sha256(self):
        return sha256(self.text.encode("utf-8")).hexdigest()


def _message_record(message: ChatMessage) -> dict:
    record = {"content": message.content, "role": message.role}
    if message.tool_call is not None:
        record["tool_call"] = {
            "arguments": _strict_json_object(message.tool_call.arguments),
            "call_id": message.tool_call.call_id,
            "name": message.tool_call.name,
        }
    if message.tool_call_id is not None:
        record["tool_call_id"] = message.tool_call_id
    return record


def render_conversation(conversation: Conversation) -> RenderedConversation:
    """Serialize one conversation as canonical JSON Lines with assistant spans.

    JSON escaping makes user-controlled text unambiguous: content resembling role
    markers remains a string value and cannot create an additional training turn.
    """
    if not isinstance(conversation, Conversation):
        raise TypeError("conversation must be Conversation")
    parts = []
    assistant_spans = []
    offset = 0
    for message in conversation.messages:
        line = (
            json.dumps(
                _message_record(message),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        parts.append(line)
        if message.role == "assistant":
            assistant_spans.append((offset, offset + len(line)))
        offset += len(line)
    return RenderedConversation(
        conversation.conversation_id, "".join(parts), tuple(assistant_spans)
    )


def canonical_digest(value) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ConversationSplit:
    train: tuple[Conversation, ...]
    validation: tuple[Conversation, ...]

    def __post_init__(self):
        if any(
            not isinstance(partition, tuple)
            or not partition
            or any(not isinstance(row, Conversation) for row in partition)
            for partition in (self.train, self.validation)
        ):
            raise SFTDataError("chat partitions require nonempty conversation tuples")
        conversations = self.train + self.validation
        if len({row.conversation_id for row in conversations}) != len(conversations):
            raise SFTDataError("conversation ids must be unique across partitions")
        rendered = tuple(render_conversation(row) for row in conversations)
        if len({row.text for row in rendered}) != len(rendered):
            raise SFTDataError("duplicate rendered conversations are not allowed")
        codec = self.codec()
        for row in rendered[len(self.train) :]:
            try:
                codec.encode(row.text)
            except ValueError as exc:
                raise SFTDataError(
                    "validation contains a character absent from training"
                ) from exc

    def codec(self) -> CharacterCodec:
        return CharacterCodec.from_text(
            "".join(render_conversation(row).text for row in self.train)
        )

    def evidence(self) -> dict:
        result = {}
        for name in ("train", "validation"):
            rendered = tuple(render_conversation(row) for row in getattr(self, name))
            result[name] = {
                "conversation_ids": [row.conversation_id for row in rendered],
                "sha256": canonical_digest(
                    [[row.conversation_id, row.text] for row in rendered]
                ),
                "characters": sum(len(row.text) for row in rendered),
                "assistant_characters": sum(
                    end - start
                    for row in rendered
                    for start, end in row.assistant_spans
                ),
            }
        result["vocabulary"] = list(self.codec().tokens)
        return result


CHAT_FIXTURE_ID = "authored-weather-tools-v1"


def _plain(conversation_id, place, value):
    return Conversation(
        conversation_id,
        (
            ChatMessage("system", "Use only supplied readings."),
            ChatMessage("user", f"What is the {place} reading?"),
            ChatMessage("assistant", f"The {place} reading is {value} C."),
            ChatMessage("user", "Is that below 20 C?"),
            ChatMessage("assistant", "Yes." if value < 20 else "No."),
        ),
    )


def _tool(conversation_id, place, value):
    call = ToolCall(
        f"call_{conversation_id.replace('-', '_')}",
        "weather.lookup",
        json.dumps({"place": place}, separators=(",", ":")),
    )
    return Conversation(
        conversation_id,
        (
            ChatMessage("system", "Use weather.lookup before answering."),
            ChatMessage("user", f"Report the {place} reading."),
            ChatMessage("assistant", "", tool_call=call),
            ChatMessage(
                "tool",
                json.dumps({"reading_c": value}, separators=(",", ":")),
                tool_call_id=call.call_id,
            ),
            ChatMessage("assistant", f"The {place} reading is {value} C."),
        ),
    )


def authored_chat_fixture() -> ConversationSplit:
    """Return deterministic artificial chats with no downloaded or personal data."""
    train = (
        _plain("plain-east-18", "east", 18),
        _plain("plain-west-21", "west", 21),
        _plain("plain-north-17", "north", 17),
        _plain("plain-south-22", "south", 22),
        _tool("tool-east-18", "east", 18),
        _tool("tool-west-21", "west", 21),
        _tool("tool-north-17", "north", 17),
        _tool("tool-south-22", "south", 22),
    )
    validation = (
        _plain("heldout-east-21", "east", 21),
        _plain("heldout-south-17", "south", 17),
        _tool("heldout-tool-west-18", "west", 18),
        _tool("heldout-tool-north-22", "north", 22),
    )
    return ConversationSplit(train, validation)


def bounded_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise SFTDataError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class SFTWindow:
    conversation_id: str
    start: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    target_mask: tuple[bool, ...]

    def __post_init__(self):
        width = len(self.inputs)
        if (
            type(self.start) is not int
            or self.start < 0
            or width == 0
            or len(self.targets) != width
            or len(self.target_mask) != width
            or any(type(value) is not bool for value in self.target_mask)
            or not any(self.target_mask)
        ):
            raise SFTDataError(
                "SFT windows require aligned tokens and at least one assistant target"
            )

    @property
    def supervised_tokens(self):
        return sum(self.target_mask)


def _assistant_positions(rendered: RenderedConversation) -> frozenset[int]:
    return frozenset(
        index for start, end in rendered.assistant_spans for index in range(start, end)
    )


def sft_training_batches(conversations, codec, *, block_size, batch_size, steps, seed):
    """Sample assistant-bearing document-local windows with a private RNG."""
    bounded_integer(block_size, "block_size", 1, 256)
    bounded_integer(batch_size, "batch_size", 1, 16)
    bounded_integer(steps, "steps", 1, 128)
    bounded_integer(seed, "seed", 0, 2**32 - 1)
    if (
        not isinstance(conversations, tuple)
        or not conversations
        or any(not isinstance(row, Conversation) for row in conversations)
        or len({row.conversation_id for row in conversations}) != len(conversations)
    ):
        raise SFTDataError("training requires unique conversation tuples")
    if not isinstance(codec, CharacterCodec):
        raise SFTDataError("training requires a CharacterCodec")
    candidates = []
    for conversation in conversations:
        rendered = render_conversation(conversation)
        tokens = codec.encode(rendered.text)
        positions = _assistant_positions(rendered)
        if len(tokens) <= block_size:
            raise SFTDataError(
                "every training conversation needs a full block and target"
            )
        for start in range(len(tokens) - block_size):
            mask = tuple(
                target_index in positions
                for target_index in range(start + 1, start + block_size + 1)
            )
            if any(mask):
                candidates.append(
                    SFTWindow(
                        conversation.conversation_id,
                        start,
                        tokens[start : start + block_size],
                        tokens[start + 1 : start + block_size + 1],
                        mask,
                    )
                )
    if not candidates:
        raise SFTDataError("training split has no assistant targets")
    rng = random.Random(seed)
    return tuple(
        tuple(rng.choice(candidates) for _ in range(batch_size)) for _ in range(steps)
    )


def sft_schedule_evidence(batches):
    if (
        not isinstance(batches, tuple)
        or not batches
        or any(
            not isinstance(batch, tuple)
            or not batch
            or any(not isinstance(row, SFTWindow) for row in batch)
            for batch in batches
        )
    ):
        raise SFTDataError("schedule evidence requires nonempty SFT batches")
    return {
        "steps": len(batches),
        "windows": sum(len(batch) for batch in batches),
        "input_tokens": sum(len(row.inputs) for batch in batches for row in batch),
        "supervised_tokens": sum(
            row.supervised_tokens for batch in batches for row in batch
        ),
        "sha256": canonical_digest(
            [
                [
                    [
                        row.conversation_id,
                        row.start,
                        row.inputs,
                        row.targets,
                        row.target_mask,
                    ]
                    for row in batch
                ]
                for batch in batches
            ]
        ),
    }


@dataclass(frozen=True)
class SFTTarget:
    conversation_id: str
    target_index: int
    context: tuple[int, ...]
    target: int

    def __post_init__(self):
        if (
            type(self.target_index) is not int
            or self.target_index <= 0
            or not isinstance(self.context, tuple)
            or not self.context
            or any(type(token) is not int for token in self.context)
            or type(self.target) is not int
        ):
            raise SFTDataError("SFT targets require a nonempty causal context")


def sft_validation_targets(conversations, codec, *, block_size):
    """Enumerate each assistant character once with bounded local context."""
    bounded_integer(block_size, "block_size", 1, 256)
    if (
        not isinstance(conversations, tuple)
        or not conversations
        or any(not isinstance(row, Conversation) for row in conversations)
        or len({row.conversation_id for row in conversations}) != len(conversations)
    ):
        raise SFTDataError("validation requires unique conversation tuples")
    if not isinstance(codec, CharacterCodec):
        raise SFTDataError("validation requires a CharacterCodec")
    targets = []
    for conversation in conversations:
        rendered = render_conversation(conversation)
        tokens = codec.encode(rendered.text)
        for start, end in rendered.assistant_spans:
            for index in range(start, end):
                if index == 0:
                    raise SFTDataError("assistant targets require preceding context")
                targets.append(
                    SFTTarget(
                        conversation.conversation_id,
                        index,
                        tokens[max(0, index - block_size) : index],
                        tokens[index],
                    )
                )
    if not targets:
        raise SFTDataError("validation split has no assistant targets")
    return tuple(targets)


def sft_validation_evidence(conversations, codec, *, block_size):
    targets = sft_validation_targets(conversations, codec, block_size=block_size)
    per_conversation = []
    for conversation in conversations:
        count = sum(
            row.conversation_id == conversation.conversation_id for row in targets
        )
        per_conversation.append(
            {"conversation_id": conversation.conversation_id, "target_tokens": count}
        )
    return {
        "target_tokens": len(targets),
        "per_conversation": per_conversation,
        "sha256": canonical_digest(
            [
                [row.conversation_id, row.target_index, row.context, row.target]
                for row in targets
            ]
        ),
        "policy": "assistant-json-lines-only; one-last-logit-per-character; conversation-local context",
    }
