"""Validated public-safe preference data for bounded alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
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


def canonical_digest(value) -> str:
    """Hash JSON-compatible evidence using one stable public representation."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise PreferenceDataError(f"value is not canonical JSON: {exc}") from exc
    return sha256(encoded).hexdigest()


def _message_line(message: ChatMessage) -> str:
    return (
        json.dumps(
            {"content": message.content, "role": message.role},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


@dataclass(frozen=True)
class RenderedCandidate:
    pair_id: str
    label: str
    prompt_text: str
    response_text: str

    def __post_init__(self):
        if self.label not in {"chosen", "rejected"}:
            raise PreferenceDataError("candidate label must be chosen or rejected")
        if not self.prompt_text or not self.response_text:
            raise PreferenceDataError("rendered candidate sections must be nonempty")
        if not self.prompt_text.endswith("\n") or not self.response_text.endswith("\n"):
            raise PreferenceDataError("rendered candidate sections must end in newline")

    @property
    def text(self):
        return self.prompt_text + self.response_text

    @property
    def response_span(self):
        return (len(self.prompt_text), len(self.text))

    @property
    def sha256(self):
        return sha256(self.text.encode("utf-8")).hexdigest()


def render_candidate(pair: PreferencePair, label: str) -> RenderedCandidate:
    """Render one branch as injection-safe canonical JSON Lines."""
    if not isinstance(pair, PreferencePair):
        raise TypeError("pair must be PreferencePair")
    if label not in {"chosen", "rejected"}:
        raise PreferenceDataError("candidate label must be chosen or rejected")
    prompt = "".join(_message_line(message) for message in pair.prompt)
    return RenderedCandidate(
        pair_id=pair.pair_id,
        label=label,
        prompt_text=prompt,
        response_text=_message_line(getattr(pair, label)),
    )


def pair_evidence(pair: PreferencePair) -> dict:
    """Bind prompt and both labeled candidates without exposing mutable objects."""
    if not isinstance(pair, PreferencePair):
        raise TypeError("pair must be PreferencePair")
    chosen = render_candidate(pair, "chosen")
    rejected = render_candidate(pair, "rejected")
    return {
        "pair_id": pair.pair_id,
        "group_id": pair.group_id,
        "prompt_sha256": sha256(chosen.prompt_text.encode("utf-8")).hexdigest(),
        "chosen_sha256": chosen.sha256,
        "rejected_sha256": rejected.sha256,
        "chosen_characters": len(chosen.response_text),
        "rejected_characters": len(rejected.response_text),
    }


PREFERENCE_FIXTURE_ID = "authored-grounded-readings-v1"


@dataclass(frozen=True)
class PreferenceDataset:
    pairs: tuple[PreferencePair, ...]

    def __post_init__(self):
        if (
            not isinstance(self.pairs, tuple)
            or not 6 <= len(self.pairs) <= 64
            or any(not isinstance(pair, PreferencePair) for pair in self.pairs)
        ):
            raise PreferenceDataError("dataset requires six to 64 preference pairs")
        if len({pair.pair_id for pair in self.pairs}) != len(self.pairs):
            raise PreferenceDataError("preference pair ids must be unique")
        group_counts = {
            group: sum(pair.group_id == group for pair in self.pairs)
            for group in {pair.group_id for pair in self.pairs}
        }
        if len(group_counts) < 3 or any(count < 2 for count in group_counts.values()):
            raise PreferenceDataError(
                "dataset requires at least three groups with two pairs each"
            )
        branches = [
            render_candidate(pair, label).text
            for pair in self.pairs
            for label in ("chosen", "rejected")
        ]
        if len(set(branches)) != len(branches):
            raise PreferenceDataError("rendered preference branches must be unique")

    def evidence(self) -> dict:
        rows = [pair_evidence(pair) for pair in self.pairs]
        return {
            "id": PREFERENCE_FIXTURE_ID,
            "pairs": len(rows),
            "groups": sorted({pair.group_id for pair in self.pairs}),
            "rows": rows,
            "sha256": canonical_digest(rows),
        }


def _reading_pair(
    place: str, value: int, *, threshold: int
) -> tuple[PreferencePair, ...]:
    group = f"reading-{place}"
    system = ChatMessage("system", "Answer only from the supplied reading.")
    report = PreferencePair(
        f"{group}-report",
        group,
        (
            system,
            ChatMessage("user", f"Evidence: {place} is {value} C. Report the reading."),
        ),
        ChatMessage("assistant", f"The {place} reading is {value} C."),
        ChatMessage("assistant", f"The {place} reading is {value + 7} C."),
    )
    below = value < threshold
    compare = PreferencePair(
        f"{group}-threshold",
        group,
        (
            system,
            ChatMessage(
                "user",
                f"Evidence: {place} is {value} C. Is it below {threshold} C?",
            ),
        ),
        ChatMessage("assistant", "Yes." if below else "No."),
        ChatMessage("assistant", "No." if below else "Yes."),
    )
    return report, compare


def authored_preference_fixture() -> PreferenceDataset:
    """Return artificial grounded choices with no downloaded or personal data."""
    pairs = tuple(
        pair
        for controls in (
            ("east", 18, 20),
            ("west", 21, 20),
            ("north", 17, 19),
            ("south", 22, 23),
            ("upper", 16, 15),
            ("lower", 24, 25),
        )
        for pair in _reading_pair(controls[0], controls[1], threshold=controls[2])
    )
    return PreferenceDataset(pairs)
