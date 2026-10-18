"""Validated public-safe preference data for bounded alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .sft_data import ChatMessage


class PreferenceDataError(ValueError):
    """Preference data violates the paired-comparison contract."""


_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


@dataclass(frozen=True)
class PreferencePair:
    """One prompt with a preferred and a rejected assistant response."""

    pair_id: str
    group_id: str
    prompt: tuple[ChatMessage, ...]
    chosen: ChatMessage
    rejected: ChatMessage

    def __post_init__(self):
        for name, value in (("pair_id", self.pair_id), ("group_id", self.group_id)):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise PreferenceDataError(
                    f"{name} must be a 1-64 character lowercase identifier"
                )
        if (
            not isinstance(self.prompt, tuple)
            or not 1 <= len(self.prompt) <= 8
            or any(not isinstance(message, ChatMessage) for message in self.prompt)
        ):
            raise PreferenceDataError("prompt must contain one to eight chat messages")
        system_count = sum(message.role == "system" for message in self.prompt)
        if system_count > 1 or (system_count and self.prompt[0].role != "system"):
            raise PreferenceDataError(
                "prompt permits at most one leading system message"
            )
        dialogue = self.prompt[1:] if system_count else self.prompt
        if not dialogue or dialogue[0].role != "user":
            raise PreferenceDataError(
                "preference prompt must start with a user message"
            )
        expected = "user"
        for message in dialogue:
            if message.role != expected or message.tool_call is not None:
                raise PreferenceDataError(
                    "preference prompts require alternating user and plain assistant messages"
                )
            expected = "assistant" if expected == "user" else "user"
        if expected != "assistant":
            raise PreferenceDataError("preference prompt must end with a user message")
        for name, response in (("chosen", self.chosen), ("rejected", self.rejected)):
            if (
                not isinstance(response, ChatMessage)
                or response.role != "assistant"
                or response.tool_call is not None
                or response.tool_call_id is not None
            ):
                raise PreferenceDataError(f"{name} must be a plain assistant message")
        if self.chosen.content == self.rejected.content:
            raise PreferenceDataError("chosen and rejected responses must differ")
