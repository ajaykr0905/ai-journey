from fractions import Fraction

import pytest

from ai_journey.gpu_ceiling_analysis import (
    FINDINGS,
    NO_CLEAR_PATTERN,
    NO_SINGLE_PATTERN,
    REDUCED_BY_PADDING,
    REMOVED_BY_PADDING,
    bandwidth_ceiling,
    bank_analysis,
    fit_fixed_cost,
    fma_ceiling,
    fma_scaling,
    model_conflict_degree,
    ridge_point,
)
from ai_journey.gpu_ceiling_source import GpuCeilingError

STRIDES = (1, 2, 4, 8, 16, 32)


def fma_point(chains, steps, median_ns, minimum_ns=None):
    flops = 2 * 1024 * chains * steps
    return {
        "kernel": f"fma_chains_{chains}",
        "chains": chains,
        "fma_steps": steps,
        "flops_per_call": flops,
        "calls_per_sample": 1,
        "timing": {"median_ns": median_ns, "min_ns": minimum_ns or median_ns},
        "median_ns_per_call": median_ns / 1,
        "median_gflops_per_second": flops / median_ns,
        "best_gflops_per_second": flops / (minimum_ns or median_ns),
    }


def linear_sweep(chains, fixed_ns, flops_per_ns, steps=(8, 32, 128)):
    """Points that lie exactly on time = fixed + flops / rate."""

    points = []
    for step in steps:
        flops = 2 * 1024 * chains * step
        points.append(fma_point(chains, step, fixed_ns + flops // flops_per_ns))
    return points


def stream_point(kernel, buffer_bytes, rate):
    return {
        "kernel": kernel,
        "buffer_bytes": buffer_bytes,
        "median_gb_per_second": rate,
        "best_gb_per_second": rate * 1.1,
    }


def bank_points(unpadded, padded, calls=1):
    points = []
    for words, times in ((32, unpadded), (33, padded)):
        for stride, median in zip(STRIDES, times):
            points.append(
                {
                    "row_words": words,
                    "stride": stride,
                    "calls_per_sample": calls,
                    "timing": {"median_ns": median},
                    "median_ps_per_access": median * 1000 / 4096,
                }
            )
    return points


def test_the_fit_recovers_an_exact_line_in_rational_arithmetic():
    slope, fixed = fit_fixed_cost(
        [1000, 2000, 4000], [Fraction(300), Fraction(500), Fraction(900)]
    )

    assert (slope, fixed) == (Fraction(1, 5), Fraction(100))
    flat, constant = fit_fixed_cost([1, 2, 3], [Fraction(7)] * 3)
    assert (flat, constant) == (None, Fraction(7))
    with pytest.raises(GpuCeilingError, match="two or more distinct"):
        fit_fixed_cost([5, 5], [Fraction(1), Fraction(2)])


def test_scaling_reports_the_fixed_cost_and_the_time_ratios_per_chain_count():
    points = linear_sweep(1, 1000, 4) + linear_sweep(4, 2000, 8)

    (one, four) = fma_scaling(points)

    assert one["chains"] == 1 and four["chains"] == 4
    assert one["marginal_gflops_per_second"] == 4.0
    assert one["fixed_ns_per_call"] == 1000.0
    assert one["fixed_share_at_largest"] == 1000 / (1000 + 2 * 1024 * 128 // 4)
    assert [step["work_ratio_to_previous"] for step in one["points"]] == [
        None,
        4.0,
        4.0,
    ]
    assert one["points"][0]["time_ratio_to_previous"] is None
    assert one["points"][2]["time_ratio_to_previous"] == (1000 + 65536) / (1000 + 16384)
    assert four["marginal_gflops_per_second"] == 8.0


def test_the_ceiling_is_the_fastest_point_and_scales_only_with_a_small_fixed_cost():
    small = linear_sweep(8, 100, 4, steps=(8, 64, 512))
    ceiling = fma_ceiling(small + linear_sweep(1, 100, 1))

    assert (ceiling["kernel"], ceiling["fma_steps"]) == ("fma_chains_8", 512)
    assert ceiling["gflops_per_second"] == small[-1]["median_gflops_per_second"]
    assert (
        ceiling["fixed_share_of_ceiling_call"] == 100 / small[-1]["timing"]["median_ns"]
    )
    assert ceiling["scales_with_work"] is True
    overhead = fma_ceiling(linear_sweep(8, 10**6, 5, steps=(8, 16, 24)))
    assert overhead["fixed_share_of_ceiling_call"] > 0.05
    assert overhead["scales_with_work"] is False
    flat = fma_ceiling([fma_point(2, step, 5000) for step in (8, 16, 24)])
    assert flat["scales_with_work"] is False


def test_the_bandwidth_ceiling_takes_the_better_kernel_at_the_largest_buffer():
    points = [
        stream_point("stream_copy", 1 << 24, 190.0),
        stream_point("stream_copy", 1 << 28, 105.0),
        stream_point("stream_read_reduce", 1 << 24, 200.0),
        stream_point("stream_read_reduce", 1 << 28, 118.0),
    ]

    ceiling = bandwidth_ceiling(points)

    assert ceiling == {
        "gb_per_second": 118.0,
        "best_sample_gb_per_second": 118.0 * 1.1,
        "kernel": "stream_read_reduce",
        "buffer_bytes": 1 << 28,
        "copy_gb_per_second": 105.0,
        "read_reduce_gb_per_second": 118.0,
    }
    with pytest.raises(GpuCeilingError, match="one copy and one read-reduce"):
        bandwidth_ceiling(points[:2])
    assert ridge_point({"gflops_per_second": 4720.0}, ceiling) == 4720.0 / 118.0


def test_the_bank_model_degrees_follow_the_32_bank_rule():
    assert [model_conflict_degree(stride, 32) for stride in STRIDES] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    assert [model_conflict_degree(stride, 33) for stride in STRIDES] == [1] * 6
    assert model_conflict_degree(3, 32) == 1


@pytest.mark.parametrize(
    ("unpadded", "padded", "finding"),
    [
        # Times like the development host's: padding helps but does not flatten.
        (
            [1410, 2930, 4800, 7290, 13380, 26580],
            [2200, 2850, 4880, 5170, 6110, 6210],
            REDUCED_BY_PADDING,
        ),
        (
            [1000, 2000, 4000, 8000, 16000, 32000],
            [1000, 1010, 990, 1020, 1000, 1010],
            REMOVED_BY_PADDING,
        ),
        ([1000, 1050, 990, 1100, 1020, 1000], [1000] * 6, NO_CLEAR_PATTERN),
        ([1000, 3000, 2000, 4000, 8000, 9000], [1000] * 6, NO_SINGLE_PATTERN),
        # Padding flattens the stride slowdown here, at eight times the cost.
        ([1000, 2000, 4000, 8000, 16000, 32000], [8000] * 6, REDUCED_BY_PADDING),
        ([1000, 2000, 4000, 8000, 16000, 32000], [40000] * 6, NO_SINGLE_PATTERN),
    ],
)
def test_the_bank_finding_says_only_what_the_times_show(unpadded, padded, finding):
    analysis = bank_analysis(bank_points(unpadded, padded))

    assert analysis["finding"] == finding and finding in FINDINGS
    rows = analysis["rows"]
    assert [row["stride"] for row in rows] == list(STRIDES)
    assert rows[0]["unpadded_relative_to_stride_1"] == 1.0
    assert rows[-1]["unpadded_relative_to_stride_1"] == unpadded[-1] / unpadded[0]
    assert rows[-1]["padded_relative_to_own_stride_1"] == padded[-1] / padded[0]
    assert rows[-1]["padded_relative_to_unpadded_stride_1"] == padded[-1] / unpadded[0]
    assert rows[-1]["padding_speedup"] == unpadded[-1] / padded[-1]
    assert analysis["largest_padded_ratio"] == max(padded) / unpadded[0]


def test_bank_ratios_use_exact_times_per_call_and_need_matching_strides():
    batched = bank_analysis(bank_points([3, 6, 12, 24, 48, 96], [3] * 6, calls=3))

    assert batched["rows"][1]["unpadded_relative_to_stride_1"] == 2.0
    broken = bank_points([1] * 6, [1] * 6)
    broken[6]["stride"] = 3
    with pytest.raises(GpuCeilingError, match="same strides"):
        bank_analysis(broken)
