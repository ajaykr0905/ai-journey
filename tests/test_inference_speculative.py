from fractions import Fraction

import pytest

from ai_journey.inference_kv import InferenceCapacityError
from ai_journey.inference_speculative import SpeculativePlan


def test_speculative_plan_uses_exact_truncated_geometric_expectation():
    plan = SpeculativePlan(4, 4, 5)

    assert plan.acceptance_probability == Fraction(4, 5)
    assert plan.expected_accepted_draft_tokens == Fraction(1476, 625)
    assert plan.expected_output_tokens_per_target_pass == Fraction(2101, 625)


def test_perfect_acceptance_returns_all_drafts_plus_one_target_token():
    plan = SpeculativePlan(8, 1, 1)

    assert plan.expected_accepted_draft_tokens == 8
    assert plan.expected_output_tokens_per_target_pass == 9


def test_speculative_plan_rejects_probability_above_one():
    with pytest.raises(InferenceCapacityError, match="cannot exceed"):
        SpeculativePlan(4, 6, 5)


def test_speculative_speedup_charges_draft_and_target_work():
    plan = SpeculativePlan(4, 4, 5)

    assert plan.expected_speedup_bound(
        draft_cost_numerator=1, draft_cost_denominator=10
    ) == Fraction(2101, 875)


def test_break_even_search_uses_the_declared_probability_grid():
    plan = SpeculativePlan(4, 1, 10)

    assert plan.minimum_acceptance_for_break_even(
        draft_cost_numerator=1, draft_cost_denominator=10
    ) == Fraction(3, 10)
