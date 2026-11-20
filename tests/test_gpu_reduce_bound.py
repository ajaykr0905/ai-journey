import math
from fractions import Fraction

import numpy
import pytest

from ai_journey.gpu_reduce_bound import (
    PATTERNS,
    VENDOR_DEPTH,
    ExactSums,
    ReductionCheck,
    benchmark_values,
    case_values,
    combine_depth,
    combine_launch,
    exact_sums,
    gamma,
    partial_count,
    pass_one_launch,
    prefix_sums,
    summation_depth,
    thread_chain_depth,
)
from ai_journey.gpu_reduce_source import (
    COMBINE,
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    STEP_KERNELS,
    GpuReduceError,
)

# Hand counts of D: thread chain + group tree + combine chain + combine tree.
DEPTHS = {
    1: {R0: 18, R1: 18, R2: 19, R3: 19, R4: 49, R5: 48},
    33: {R0: 18, R1: 18, R2: 19, R3: 19, R4: 49, R5: 48},
    1000003: {R0: 21, R1: 21, R2: 20, R3: 20, R4: 50, R5: 51},
    1 << 26: {R0: 273, R1: 273, R2: 146, R3: 146, R4: 176, R5: 303},
}


def fraction_sum(values):
    return sum((Fraction(float(value)) for value in values), Fraction(0))


def sequential_replay(values):
    """R1's order on the CPU in float32: 256-value trees, then the combine."""

    groups = -(-values.size // 256)
    tree = numpy.zeros(groups * 256, dtype=numpy.float32)
    tree[: values.size] = values
    tree = tree.reshape(groups, 256)
    width = 256
    while width > 1:
        width //= 2
        tree[:, :width] += tree[:, width : 2 * width]
    partials = tree[:, 0]
    chains = numpy.zeros(1024 * -(-groups // 1024), dtype=numpy.float32)
    chains[:groups] = partials
    threads = numpy.zeros(1024, dtype=numpy.float32)
    for row in chains.reshape(-1, 1024):
        threads += row
    width = 1024
    while width > 1:
        width //= 2
        threads[:width] += threads[width : 2 * width]
    return float(threads[0])


@pytest.mark.parametrize("pattern", PATTERNS)
def test_exact_sums_match_fractions_and_the_correctly_rounded_fsum(pattern):
    values = case_values(78, 3, pattern, 4099)

    sums = exact_sums(values)

    assert sums.total == fraction_sum(values)
    assert sums.magnitude == fraction_sum(numpy.abs(values))
    assert sums.reference == math.fsum(values.astype(float).tolist())
    assert sums.sum_abs == math.fsum(numpy.abs(values).astype(float).tolist())
    assert values.dtype == numpy.float32 and values.size == 4099
    assert numpy.array_equal(values, case_values(78, 3, pattern, 4099))


def test_each_pattern_draws_its_declared_values():
    mixed = case_values(78, 1, "mixed", 10000).astype(float)
    exponents = numpy.frexp(numpy.abs(mixed))[1] - 1
    assert exponents.min() == -24 and exponents.max() == 24
    halves = case_values(78, 2, "halves", 10000)
    assert set((halves * 2).astype(int).tolist()) == set(range(-8, 9))
    positive = case_values(78, 3, "positive", 10000)
    assert positive.min() >= 0.0 and positive.max() < 1.0
    assert set(case_values(78, 4, "ones", 7).tolist()) == {1.0}
    assert not numpy.array_equal(
        case_values(78, 5, "normal", 64), case_values(78, 6, "normal", 64)
    )
    with pytest.raises(GpuReduceError, match="pattern must be one of"):
        case_values(78, 1, "uniform", 4)
    with pytest.raises(GpuReduceError, match="case index"):
        case_values(78, 64, "normal", 4)
    timing = benchmark_values(78, 1 << 12)
    assert timing.dtype == numpy.float32
    assert not numpy.array_equal(timing[:64], case_values(78, 0, "normal", 64))


def test_exact_cases_are_named_by_the_grid_rule():
    ones = exact_sums(case_values(78, 1, "ones", 1000003))
    assert (ones.grid_exponent, ones.exact_in_any_order) == (0, True)
    assert ones.total == 1000003
    halves = exact_sums(case_values(78, 2, "halves", 1000003))
    assert (halves.grid_exponent, halves.exact_in_any_order) == (-1, True)
    assert exact_sums(numpy.ones(1 << 24, dtype=numpy.float32)).exact_in_any_order
    assert not exact_sums(
        numpy.ones((1 << 24) + 1, dtype=numpy.float32)
    ).exact_in_any_order
    single = exact_sums(case_values(78, 3, "normal", 1))
    assert single.exact_in_any_order
    assert not exact_sums(case_values(78, 3, "normal", 33)).exact_in_any_order
    zeros = exact_sums(numpy.zeros(5, dtype=numpy.float32))
    assert (zeros.grid_exponent, zeros.exact_in_any_order) == (None, True)
    assert zeros.to_mapping() == {
        "reference": 0.0,
        "sum_abs": 0.0,
        "grid_exponent": None,
        "exact_in_any_order": True,
    }


def test_inputs_that_could_make_a_subnormal_partial_sum_are_refused():
    tiny = numpy.array([1.0, 2.0**-127], dtype=numpy.float32)
    with pytest.raises(GpuReduceError, match="finer than 2\\^-126"):
        exact_sums(tiny)
    for values, message in (
        (numpy.array([1.0, numpy.inf], dtype=numpy.float32), "finite"),
        (numpy.ones(4, dtype=numpy.float64), "1D float32 array"),
        (numpy.ones((2, 2), dtype=numpy.float32), "1D float32 array"),
    ):
        with pytest.raises(GpuReduceError, match=message):
            exact_sums(values)
    with pytest.raises(GpuReduceError, match="cannot be below"):
        ExactSums(Fraction(3), Fraction(2), 0)
    with pytest.raises(GpuReduceError, match="only an all-zero input"):
        ExactSums(Fraction(1), Fraction(1), None)


def test_prefix_sums_equal_the_sums_of_each_prefix():
    values = case_values(78, 7, "mixed", 300007)

    sums = prefix_sums(values, (300007, 1, 4096, 4097))

    assert list(sums) == [1, 4096, 4097, 300007]
    for size, entry in sums.items():
        assert entry == exact_sums(values[:size])
    with pytest.raises(GpuReduceError, match="each prefix size"):
        prefix_sums(values, (300008,))


@pytest.mark.parametrize("n", sorted(DEPTHS))
def test_each_order_has_its_hand_counted_depth(n):
    assert {name: summation_depth(name, n) for name in STEP_KERNELS} == DEPTHS[n]


def test_launches_cover_n_in_whole_groups_and_fix_the_partial_count():
    n = 1000003
    assert pass_one_launch(R0, n).to_mapping()["groups"] == [3907]
    assert pass_one_launch(R0, n).threads == (3907 * 256,)
    assert [partial_count(name, n) for name in STEP_KERNELS] == [
        3907,
        3907,
        1954,
        1954,
        1954,
        1024,
    ]
    assert pass_one_launch(R5, 1).threads == (262144,)
    assert combine_launch().to_mapping()["groups"] == [1]
    assert combine_launch().group_size == (1024,)
    assert [thread_chain_depth(name, n) for name in (R0, R2, R5)] == [0, 1, 3]
    assert [combine_depth(count) for count in (1, 1024, 1025, 1 << 18)] == [
        10,
        10,
        11,
        265,
    ]
    with pytest.raises(GpuReduceError, match="not a step kernel"):
        pass_one_launch(COMBINE, n)


def test_gamma_and_the_bound_are_exact_rationals():
    assert gamma(0) == 0
    assert gamma(18) == Fraction(18, (1 << 24) - 18)
    assert float(gamma(VENDOR_DEPTH)) == pytest.approx(4096 / ((1 << 24) - 4096))
    sums = exact_sums(case_values(78, 1, "normal", 33))
    near = float(numpy.float32(sums.reference))
    check = ReductionCheck(near, 18, sums)
    assert check.within and check.error <= check.bound
    assert check.bound == gamma(18) * sums.magnitude
    assert check.to_mapping() == {
        "output": near,
        "depth": 18,
        "error": float(abs(Fraction(near) - sums.total)),
        "bound": float(gamma(18) * sums.magnitude),
        "bound_ratio": float(abs(Fraction(near) - sums.total) / check.bound),
        "exact_result": Fraction(near) == sums.total,
    }
    far = float(numpy.float32(sums.reference + 2 * float(check.bound)))
    assert not ReductionCheck(far, 18, sums).within


def test_an_exact_case_must_hold_the_exact_sum_bit_for_bit():
    sums = exact_sums(case_values(78, 1, "ones", 1000003))
    assert ReductionCheck(1000003.0, 21, sums).within
    off = float(numpy.nextafter(numpy.float32(1000003.0), numpy.float32(0)))
    assert ReductionCheck(off, 21, sums).bound_ratio < 1.0
    assert not ReductionCheck(off, 21, sums).within
    for output, message in ((math.nan, "finite float"), (0.1, "float32 value")):
        with pytest.raises(GpuReduceError, match=message):
            ReductionCheck(output, 21, sums)


def test_the_bound_tells_a_tree_order_from_a_serial_float32_loop():
    values = case_values(78, 9, "positive", 1000003)
    sums = exact_sums(values)
    depth = summation_depth(R1, values.size)

    tree = ReductionCheck(sequential_replay(values), depth, sums)
    serial = float(numpy.cumsum(values, dtype=numpy.float32)[-1])
    loop = ReductionCheck(serial, depth, sums)

    assert tree.within and tree.bound_ratio < 0.1
    assert not loop.within and loop.bound_ratio > 10
    # Any order passes the bound of its own depth: n - 1 roundings here.
    assert ReductionCheck(serial, values.size - 1, sums).within
