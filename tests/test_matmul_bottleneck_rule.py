import dataclasses
from pathlib import Path

import pytest

from ai_journey.matmul_bottleneck_config import MatmulBottleneckError, load_config
from ai_journey.matmul_bottleneck_rule import (
    BOTTLENECKS,
    COMPUTE_BOUND,
    DISPATCH_BOUND,
    INDETERMINATE,
    LATENCY_BOUND,
    MEMORY_BOUND,
    REASONS,
    PointInputs,
    classify,
)

ROOT = Path(__file__).resolve().parents[1]
RULE = load_config(ROOT / "config/day-75-matmul-bottleneck.json").rule
# A large naive point: steady 5.7 ms per call, one call 5.9 ms on the host.
BASE = PointInputs(
    steady_ns_per_call=5_712_900.0,
    rate_gflops_per_second=375.9,
    gpu_ns=5_648_875,
    host_ns=5_937_709,
    dispatch_ns=122_167,
    fma_ceiling_gflops_per_second=4757.3,
    fma_scales_with_work=True,
    bandwidth_roof_gflops_per_second=17_512.0,
)


def point(**changes):
    return dataclasses.replace(BASE, **changes)


def test_a_large_naive_point_sits_below_both_roofs_with_its_numbers():
    decision = classify(BASE, RULE)
    mapping = decision.to_mapping()

    assert (decision.bottleneck, decision.reason) == (LATENCY_BOUND, "below_both_roofs")
    numbers = mapping["numbers"]
    assert numbers["overhead_share"] == 1.0 - 5_712_900.0 / 5_937_709
    assert numbers["host_outside_gpu_share"] == 1.0 - 5_648_875 / 5_937_709
    assert numbers["dispatch_share"] == 122_167 / 5_937_709
    assert numbers["gpu_excess_share"] == (5_648_875 - 5_712_900.0) / 5_937_709
    assert numbers["gpu_to_steady_ratio"] == 5_648_875 / 5_712_900.0
    # The ideal-reuse roof is above the FMA ceiling, so the FMA ceiling is lower.
    assert numbers["lower_roof_share"] == numbers["fma_ceiling_share"] == 375.9 / 4757.3
    assert mapping["deciding"] == {
        "quantity": "lower_roof_share",
        "value": 375.9 / 4757.3,
        "limit": 0.7,
    }
    assert mapping["explanation"] == REASONS["below_both_roofs"][1]


@pytest.mark.parametrize(
    ("changes", "reason", "name"),
    [
        ({"dispatch_ns": 5_937_710}, "dispatch_time_above_host_time", INDETERMINATE),
        ({"gpu_ns": 6_300_000}, "gpu_time_above_host_time", INDETERMINATE),
        (
            {"steady_ns_per_call": 6_300_000.0},
            "steady_time_above_host_time",
            INDETERMINATE,
        ),
        (
            {"steady_ns_per_call": 20_100.0, "host_ns": 460_125, "gpu_ns": 249_125},
            "one_call_overhead_dominates",
            DISPATCH_BOUND,
        ),
        ({"gpu_ns": 3_800_000}, "gpu_time_disagrees_with_steady_time", INDETERMINATE),
        # A GPU time 1.52 times the steady time per call, the other way round.
        (
            {"gpu_ns": 5_937_709, "steady_ns_per_call": 3_900_000.0},
            "gpu_time_disagrees_with_steady_time",
            INDETERMINATE,
        ),
        ({"rate_gflops_per_second": 5100.0}, "rate_above_fma_ceiling", INDETERMINATE),
        (
            {"rate_gflops_per_second": 30.0, "bandwidth_roof_gflops_per_second": 25.0},
            "rate_above_bandwidth_roof",
            INDETERMINATE,
        ),
        (
            {"rate_gflops_per_second": 18.0, "bandwidth_roof_gflops_per_second": 25.0},
            "near_bandwidth_roof",
            MEMORY_BOUND,
        ),
        ({"rate_gflops_per_second": 4000.0}, "near_fma_ceiling", COMPUTE_BOUND),
        (
            {"rate_gflops_per_second": 4000.0, "fma_scales_with_work": False},
            "fma_ceiling_does_not_scale_with_work",
            INDETERMINATE,
        ),
    ],
)
def test_each_step_of_the_rule_names_its_reason(changes, reason, name):
    decision = classify(point(**changes), RULE)

    assert (decision.reason, decision.bottleneck) == (reason, name)
    assert REASONS[reason][0] == name


def test_the_overhead_splits_into_host_and_gpu_parts():
    # One call at 512 x 512 x 512: most of the overhead is inside the GPU timeline.
    decision = classify(
        point(steady_ns_per_call=1_655_200.0, host_ns=3_647_916, gpu_ns=3_215_583),
        RULE,
    )
    numbers = decision.numbers

    assert decision.bottleneck == DISPATCH_BOUND
    assert numbers["overhead_share"] == pytest.approx(
        numbers["host_outside_gpu_share"] + numbers["gpu_excess_share"]
    )
    assert numbers["gpu_excess_share"] > numbers["host_outside_gpu_share"]


def test_the_first_step_that_applies_wins():
    # Overhead dominates and the GPU time also disagrees: dispatch comes first.
    tiny = point(steady_ns_per_call=20_100.0, host_ns=460_125, gpu_ns=249_125)
    assert classify(tiny, RULE).bottleneck == DISPATCH_BOUND
    # A contradiction comes before the overhead step.
    contradiction = point(steady_ns_per_call=20_100.0, host_ns=460_125, gpu_ns=500_000)
    assert classify(contradiction, RULE).reason == "gpu_time_above_host_time"


def test_thresholds_are_inclusive_where_the_rule_says_at_least():
    host = 1_000_000
    half = point(steady_ns_per_call=500_000.0, host_ns=host, gpu_ns=500_000)
    assert classify(half, RULE).bottleneck == DISPATCH_BOUND
    just_under = point(steady_ns_per_call=500_001.0, host_ns=host, gpu_ns=500_001)
    assert classify(just_under, RULE).bottleneck == LATENCY_BOUND
    # 0.7 of the FMA ceiling is about 3330.11 GFLOP/s.
    assert classify(point(rate_gflops_per_second=3330.0), RULE).bottleneck == (
        LATENCY_BOUND
    )
    assert classify(point(rate_gflops_per_second=3330.2), RULE).bottleneck == (
        COMPUTE_BOUND
    )


def test_a_rule_without_the_scaling_check_trusts_the_fma_ceiling():
    relaxed = dataclasses.replace(RULE, require_fma_scales_with_work=False)
    flagged = point(rate_gflops_per_second=4000.0, fma_scales_with_work=False)

    assert classify(flagged, relaxed).bottleneck == COMPUTE_BOUND


def test_every_reason_names_a_declared_bottleneck():
    assert {name for name, _ in REASONS.values()} == set(BOTTLENECKS)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"steady_ns_per_call": 5}, "steady_ns_per_call must be"),
        ({"rate_gflops_per_second": float("inf")}, "rate_gflops_per_second must be"),
        ({"bandwidth_roof_gflops_per_second": 0.0}, "must be a positive finite"),
        ({"gpu_ns": 0}, "gpu_ns must be a positive integer"),
        ({"host_ns": 1.0}, "host_ns must be a positive integer"),
        ({"fma_scales_with_work": 1}, "must be a boolean"),
    ],
)
def test_invalid_inputs_are_refused(changes, message):
    with pytest.raises(MatmulBottleneckError, match=message):
        point(**changes)
