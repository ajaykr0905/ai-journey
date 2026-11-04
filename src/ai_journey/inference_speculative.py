"""Exact expected-token accounting for speculative decoding plans."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .inference_kv import InferenceCapacityError, _positive_integer


@dataclass(frozen=True)
class SpeculativePlan:
    draft_tokens: int
    acceptance_numerator: int
    acceptance_denominator: int

    def __post_init__(self) -> None:
        for name in (
            "draft_tokens",
            "acceptance_numerator",
            "acceptance_denominator",
        ):
            _positive_integer(getattr(self, name), name)
        if self.acceptance_numerator > self.acceptance_denominator:
            raise InferenceCapacityError("acceptance probability cannot exceed one")

    @property
    def acceptance_probability(self) -> Fraction:
        return Fraction(self.acceptance_numerator, self.acceptance_denominator)

    @property
    def expected_accepted_draft_tokens(self) -> Fraction:
        probability = self.acceptance_probability
        return sum(
            (probability**position for position in range(1, self.draft_tokens + 1)),
            start=Fraction(0),
        )

    @property
    def expected_output_tokens_per_target_pass(self) -> Fraction:
        """Include the target correction/bonus token after draft verification."""

        return 1 + self.expected_accepted_draft_tokens
