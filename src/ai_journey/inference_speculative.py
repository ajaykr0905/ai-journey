"""Exact expected-token accounting for speculative decoding plans."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .inference_kv import InferenceCapacityError, _positive_integer


@dataclass(frozen=True)
class SpeculativePlan:
    """Declared draft length and rational acceptance rate for speculative decoding.

    The rate is a declared input, not a measurement; fractions keep every
    expectation and speedup bound exact. Nonpositive fields or a rate above
    one raise InferenceCapacityError.
    """

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
        """Return the declared per-draft-token acceptance probability exactly."""

        return Fraction(self.acceptance_numerator, self.acceptance_denominator)

    @property
    def expected_accepted_draft_tokens(self) -> Fraction:
        """Return the truncated geometric sum of p**i for i in 1..draft_tokens.

        A draft token counts only when every earlier draft token was accepted.
        """

        probability = self.acceptance_probability
        return sum(
            (probability**position for position in range(1, self.draft_tokens + 1)),
            start=Fraction(0),
        )

    @property
    def expected_output_tokens_per_target_pass(self) -> Fraction:
        """Include the target correction/bonus token after draft verification."""

        return 1 + self.expected_accepted_draft_tokens

    def expected_speedup_bound(
        self, *, draft_cost_numerator: int, draft_cost_denominator: int
    ) -> Fraction:
        """Return speedup under a declared draft-token/target-pass cost ratio."""

        _positive_integer(draft_cost_numerator, "draft_cost_numerator")
        _positive_integer(draft_cost_denominator, "draft_cost_denominator")
        ratio = Fraction(draft_cost_numerator, draft_cost_denominator)
        normalized_pass_cost = 1 + self.draft_tokens * ratio
        return self.expected_output_tokens_per_target_pass / normalized_pass_cost

    def minimum_acceptance_for_break_even(
        self, *, draft_cost_numerator: int, draft_cost_denominator: int
    ) -> Fraction | None:
        """Find the smallest rational i/denominator yielding speedup at least one."""

        _positive_integer(draft_cost_numerator, "draft_cost_numerator")
        _positive_integer(draft_cost_denominator, "draft_cost_denominator")
        for numerator in range(1, self.acceptance_denominator + 1):
            candidate = SpeculativePlan(
                self.draft_tokens, numerator, self.acceptance_denominator
            )
            if (
                candidate.expected_speedup_bound(
                    draft_cost_numerator=draft_cost_numerator,
                    draft_cost_denominator=draft_cost_denominator,
                )
                >= 1
            ):
                return candidate.acceptance_probability
        return None
