from fractions import Fraction

import pytest

from ai_journey.serving_analysis import (
    GoodputTarget,
    LatencySummary,
    ServingAnalysisError,
    batch_packing_efficiency,
    failure_fraction,
    goodput_fraction,
    nearest_rank_percentile,
    output_throughput_per_second,
    retry_amplification,
    speculative_acceptance_fraction,
    successful_cache_hit_fraction,
    successful_ttft_summary,
)
from ai_journey.serving_observation import (
    RequestObservation,
    RequestTiming,
    ServingOutcome,
)


def test_nearest_rank_percentile_is_deterministic_for_small_samples():
    values = (40, 10, 30, 20)

    assert nearest_rank_percentile(values, 50) == 20
    assert nearest_rank_percentile(values, 95) == 40
    assert nearest_rank_percentile(values, 100) == 40


@pytest.mark.parametrize("values,percentile", [((), 50), ((1,), 0), ((-1,), 50)])
def test_nearest_rank_percentile_rejects_undefined_inputs(values, percentile):
    with pytest.raises((ServingAnalysisError, TypeError)):
        nearest_rank_percentile(values, percentile)


def _observation(
    request_id,
    *,
    received_ns=0,
    started_ns=10,
    first_token_ns=20,
    completed_ns=100,
    outcome=ServingOutcome.SUCCESS,
    generated_tokens=4,
    cache_hit=False,
    retry_count=0,
):
    return RequestObservation(
        request_id=request_id,
        outcome=outcome,
        timing=RequestTiming(received_ns, started_ns, first_token_ns, completed_ns),
        prompt_tokens=16,
        generated_tokens=generated_tokens,
        cache_hit=cache_hit,
        retry_count=retry_count,
    )


def test_successful_ttft_summary_excludes_censored_requests():
    observations = (
        _observation("a", first_token_ns=10),
        _observation("b", first_token_ns=20),
        _observation(
            "c",
            first_token_ns=None,
            outcome=ServingOutcome.TIMEOUT,
            generated_tokens=0,
        ),
    )

    assert successful_ttft_summary(observations) == LatencySummary(10, 20, 20)


def test_output_throughput_uses_successful_tokens_and_full_wall_interval():
    observations = (
        _observation("a", completed_ns=1_000_000_000, generated_tokens=4),
        _observation(
            "b",
            received_ns=500_000_000,
            started_ns=500_000_010,
            first_token_ns=None,
            completed_ns=1_000_000_000,
            outcome=ServingOutcome.ERROR,
            generated_tokens=0,
        ),
    )

    assert output_throughput_per_second(observations) == Fraction(4, 1)


def test_goodput_requires_success_inside_both_latency_targets():
    observations = (
        _observation("a", first_token_ns=20, completed_ns=80),
        _observation("b", first_token_ns=60, completed_ns=80),
    )

    assert goodput_fraction(observations, GoodputTarget(50, 100)) == Fraction(1, 2)


def test_failure_fraction_counts_every_censored_terminal_outcome():
    observations = (
        _observation("a"),
        _observation("b", outcome=ServingOutcome.TIMEOUT, generated_tokens=0),
        _observation("c", outcome=ServingOutcome.OVERLOADED, generated_tokens=0),
    )

    assert failure_fraction(observations) == Fraction(2, 3)


def test_retry_amplification_includes_initial_attempts():
    observations = (
        _observation("a", retry_count=0),
        _observation("b", retry_count=2),
    )

    assert retry_amplification(observations) == Fraction(2, 1)


def test_cache_hit_fraction_excludes_failed_requests():
    observations = (
        _observation("a", cache_hit=True),
        _observation("b", cache_hit=False),
        _observation(
            "c",
            outcome=ServingOutcome.ERROR,
            generated_tokens=0,
            cache_hit=True,
        ),
    )

    assert successful_cache_hit_fraction(observations) == Fraction(1, 2)


def test_speculative_acceptance_is_exact_and_bounded():
    assert speculative_acceptance_fraction(
        accepted_tokens=8, drafted_tokens=10
    ) == Fraction(4, 5)

    with pytest.raises(ServingAnalysisError, match="exceed"):
        speculative_acceptance_fraction(accepted_tokens=11, drafted_tokens=10)


def test_batch_packing_efficiency_exposes_padding_waste():
    assert batch_packing_efficiency(
        useful_tokens=96, allocated_token_slots=128
    ) == Fraction(3, 4)

    with pytest.raises(ServingAnalysisError, match="exceed"):
        batch_packing_efficiency(useful_tokens=129, allocated_token_slots=128)
