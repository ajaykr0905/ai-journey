"""Validated public-safe preference data for bounded alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
import re

from .sft_data import ChatMessage
from .transformer_lab import CharacterCodec


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


@dataclass(frozen=True)
class PreferenceFold:
    index: int
    train: tuple[PreferencePair, ...]
    heldout: tuple[PreferencePair, ...]

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise PreferenceDataError("fold index must be a nonnegative integer")
        for name, partition in (("train", self.train), ("heldout", self.heldout)):
            if (
                not isinstance(partition, tuple)
                or not partition
                or any(not isinstance(pair, PreferencePair) for pair in partition)
            ):
                raise PreferenceDataError(f"fold {name} partition must be nonempty")
        train_ids = {pair.pair_id for pair in self.train}
        heldout_ids = {pair.pair_id for pair in self.heldout}
        train_groups = {pair.group_id for pair in self.train}
        heldout_groups = {pair.group_id for pair in self.heldout}
        if train_ids & heldout_ids or train_groups & heldout_groups:
            raise PreferenceDataError("fold train and heldout groups must be disjoint")

    def evidence(self) -> dict:
        return {
            "index": self.index,
            "train_pair_ids": [pair.pair_id for pair in self.train],
            "heldout_pair_ids": [pair.pair_id for pair in self.heldout],
            "train_groups": sorted({pair.group_id for pair in self.train}),
            "heldout_groups": sorted({pair.group_id for pair in self.heldout}),
        }


def grouped_preference_folds(
    dataset: PreferenceDataset, *, fold_count: int, seed: int
) -> tuple[PreferenceFold, ...]:
    """Split whole prompt families so related pairs never cross a fold boundary."""
    if not isinstance(dataset, PreferenceDataset):
        raise TypeError("dataset must be PreferenceDataset")
    if type(fold_count) is not int or not 2 <= fold_count <= 8:
        raise PreferenceDataError("fold_count must be an integer in [2, 8]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise PreferenceDataError("fold seed must be an integer in [0, 2**32)")
    groups = sorted({pair.group_id for pair in dataset.pairs})
    if fold_count > len(groups):
        raise PreferenceDataError("fold_count cannot exceed the number of groups")
    random.Random(seed).shuffle(groups)
    heldout_groups = [set() for _ in range(fold_count)]
    for index, group in enumerate(groups):
        heldout_groups[index % fold_count].add(group)
    folds = []
    for index, selected in enumerate(heldout_groups):
        heldout = tuple(pair for pair in dataset.pairs if pair.group_id in selected)
        train = tuple(pair for pair in dataset.pairs if pair.group_id not in selected)
        folds.append(PreferenceFold(index, train, heldout))
    if {pair.pair_id for fold in folds for pair in fold.heldout} != {
        pair.pair_id for pair in dataset.pairs
    }:
        raise PreferenceDataError("cross-validation must hold out every pair once")
    return tuple(folds)


def fold_evidence(folds: tuple[PreferenceFold, ...]) -> dict:
    if (
        not isinstance(folds, tuple)
        or len(folds) < 2
        or any(not isinstance(fold, PreferenceFold) for fold in folds)
        or tuple(fold.index for fold in folds) != tuple(range(len(folds)))
    ):
        raise PreferenceDataError("fold evidence requires ordered preference folds")
    rows = [fold.evidence() for fold in folds]
    return {"folds": rows, "sha256": canonical_digest(rows)}


def preference_codec(dataset: PreferenceDataset, *, extra_text: str = ""):
    """Build one explicit character vocabulary for every labeled branch."""
    if not isinstance(dataset, PreferenceDataset):
        raise TypeError("dataset must be PreferenceDataset")
    if not isinstance(extra_text, str):
        raise TypeError("extra_text must be text")
    text = "".join(
        render_candidate(pair, label).text
        for pair in dataset.pairs
        for label in ("chosen", "rejected")
    )
    return CharacterCodec.from_text(text + extra_text)


@dataclass(frozen=True)
class CandidateTargets:
    pair_id: str
    label: str
    contexts: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]

    def __post_init__(self):
        if self.label not in {"chosen", "rejected"}:
            raise PreferenceDataError("candidate target label is invalid")
        if (
            not isinstance(self.contexts, tuple)
            or not self.contexts
            or not isinstance(self.targets, tuple)
            or len(self.contexts) != len(self.targets)
            or any(
                not context or any(type(token) is not int for token in context)
                for context in self.contexts
            )
            or any(type(token) is not int for token in self.targets)
        ):
            raise PreferenceDataError(
                "candidate targets require aligned causal contexts"
            )

    @property
    def target_tokens(self):
        return len(self.targets)


def candidate_targets(
    pair: PreferencePair, label: str, codec: CharacterCodec, *, block_size: int
) -> CandidateTargets:
    """Enumerate each response character once with bounded prompt-local context."""
    if not isinstance(codec, CharacterCodec):
        raise TypeError("codec must be CharacterCodec")
    if type(block_size) is not int or not 1 <= block_size <= 256:
        raise PreferenceDataError("block_size must be an integer in [1, 256]")
    rendered = render_candidate(pair, label)
    tokens = codec.encode(rendered.text)
    start, end = rendered.response_span
    contexts = tuple(
        tokens[max(0, index - block_size) : index] for index in range(start, end)
    )
    targets = tuple(tokens[index] for index in range(start, end))
    return CandidateTargets(pair.pair_id, label, contexts, targets)


def target_evidence(
    pairs: tuple[PreferencePair, ...], codec: CharacterCodec, *, block_size: int
) -> dict:
    if (
        not isinstance(pairs, tuple)
        or not pairs
        or any(not isinstance(pair, PreferencePair) for pair in pairs)
    ):
        raise PreferenceDataError("target evidence requires preference pairs")
    rows = []
    for pair in pairs:
        for label in ("chosen", "rejected"):
            targets = candidate_targets(pair, label, codec, block_size=block_size)
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "label": label,
                    "target_tokens": targets.target_tokens,
                    "sha256": canonical_digest([targets.contexts, targets.targets]),
                }
            )
    return {
        "policy": "response-json-line-only; one last-logit score per character",
        "rows": rows,
        "target_tokens": sum(row["target_tokens"] for row in rows),
        "sha256": canonical_digest(rows),
    }


def preference_training_batches(
    pairs: tuple[PreferencePair, ...], *, batch_size: int, steps: int, seed: int
) -> tuple[tuple[PreferencePair, ...], ...]:
    """Sample a fixed pair schedule without consuming the caller's RNG."""
    if (
        not isinstance(pairs, tuple)
        or not pairs
        or any(not isinstance(pair, PreferencePair) for pair in pairs)
        or len({pair.pair_id for pair in pairs}) != len(pairs)
    ):
        raise PreferenceDataError("training requires unique preference pairs")
    for name, value, minimum, maximum in (
        ("batch_size", batch_size, 1, 16),
        ("steps", steps, 1, 128),
        ("seed", seed, 0, 2**32 - 1),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise PreferenceDataError(
                f"{name} must be an integer in [{minimum}, {maximum}]"
            )
    rng = random.Random(seed)
    return tuple(
        tuple(rng.choice(pairs) for _ in range(batch_size)) for _ in range(steps)
    )


def preference_schedule_evidence(
    batches: tuple[tuple[PreferencePair, ...], ...],
) -> dict:
    if (
        not isinstance(batches, tuple)
        or not batches
        or any(
            not isinstance(batch, tuple)
            or not batch
            or any(not isinstance(pair, PreferencePair) for pair in batch)
            for batch in batches
        )
    ):
        raise PreferenceDataError("schedule evidence requires preference batches")
    width = len(batches[0])
    if any(len(batch) != width for batch in batches):
        raise PreferenceDataError("preference batches must have a fixed width")
    pair_ids = [[pair.pair_id for pair in batch] for batch in batches]
    return {
        "steps": len(batches),
        "batch_size": width,
        "comparisons": len(batches) * width,
        "pair_ids": pair_ids,
        "sha256": canonical_digest(pair_ids),
    }
