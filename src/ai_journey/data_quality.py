"""Explainable text-quality admission for cleaned training documents."""

from dataclasses import dataclass

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument


@dataclass(frozen=True)
class QualityPolicy:
    """Admission thresholds; the defaults are the pinned Day 59 values.

    Lengths count code points of cleaned text and the alphanumeric fraction
    ignores whitespace. Out-of-bounds values raise TokenizerError.
    """

    min_characters: int = 20
    max_characters: int = 4_096
    min_alphanumeric_fraction: float = 0.5
    max_repeated_run: int = 8

    def __post_init__(self) -> None:
        if (
            type(self.min_characters) is not int
            or type(self.max_characters) is not int
            or not 1 <= self.min_characters <= self.max_characters <= 1_000_000
            or isinstance(self.min_alphanumeric_fraction, bool)
            or not isinstance(self.min_alphanumeric_fraction, (int, float))
            or not 0 <= self.min_alphanumeric_fraction <= 1
            or type(self.max_repeated_run) is not int
            or not 1 <= self.max_repeated_run <= 1_000
        ):
            raise TokenizerError("quality policy bounds are invalid")


@dataclass(frozen=True)
class QualityDecision:
    """Admission outcome that keeps a rejected document and every rule it failed."""

    document: CleanedDocument
    accepted: bool
    reasons: tuple[str, ...]


def assess_quality(document: CleanedDocument, policy: QualityPolicy) -> QualityDecision:
    """Apply every rule and collect all failures instead of stopping at the first.

    Reasons are too_short, too_long, low_alphanumeric_fraction and
    repeated_character_run.
    """
    if not isinstance(document, CleanedDocument) or not isinstance(
        policy, QualityPolicy
    ):
        raise TypeError("quality assessment requires cleaned text and a policy")
    reasons = []
    length = len(document.text)
    if length < policy.min_characters:
        reasons.append("too_short")
    if length > policy.max_characters:
        reasons.append("too_long")
    visible = [character for character in document.text if not character.isspace()]
    fraction = sum(character.isalnum() for character in visible) / len(visible)
    if fraction < policy.min_alphanumeric_fraction:
        reasons.append("low_alphanumeric_fraction")
    run = longest_character_run(document.text)
    if run > policy.max_repeated_run:
        reasons.append("repeated_character_run")
    return QualityDecision(document, not reasons, tuple(reasons))


def longest_character_run(text: str) -> int:
    """Longest run of one repeated character ("!!!!" is 4), for the run-length rule."""
    longest = active = 0
    previous = None
    for character in text:
        active = active + 1 if character == previous else 1
        longest = max(longest, active)
        previous = character
    return longest
