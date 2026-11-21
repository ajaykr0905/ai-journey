import math
from hashlib import sha256

import numpy
import pytest

from ai_journey.metal_fused_reference import (
    BENCHMARK_STREAM,
    CORRECTNESS_GAP,
    CORRECTNESS_STREAM,
    DISTRIBUTIONS,
    EXP_ERROR,
    EXP_PROBE_COUNT,
    EXP_PROBE_EXTREMES,
    FLOOR,
    GAP_INPUT,
    MAX_ELEMENTS,
    NORMAL,
    OFFSET_MINUS_1000,
    OFFSET_PLUS_1000,
    SENTINEL_BITS,
    UNIFORM_80,
    RowLayout,
    dense_layout,
    error_bounds,
    exp_probe_arguments,
    exp_probe_errors,
    float32_sha256,
    float64_sha256,
    naive_float32_finite,
    order_terms,
    ragged_layout,
    row_sum_errors,
    sentinel_buffer,
    softmax_inputs,
    softmax_reference,
)
from ai_journey.metal_fused_source import F1, F2, F3, U0, MetalFusedError


def test_a_gapped_layout_places_each_row_after_the_guard_of_the_last():
    layout = RowLayout((3, 1, 5), 2)

    assert layout.starts().tolist() == [0, 5, 8]
    assert layout.valid().tolist() == [0, 1, 2, 5, 8, 9, 10, 11, 12]
    assert (layout.rows, layout.elements, layout.longest) == (3, 9, 5)
    assert layout.buffer_elements == 15 and layout.group_size == 32
    assert not layout.dense and dense_layout(4, 7).dense
    assert dense_layout(4, 7).valid().tolist() == list(range(28))
    mapping = layout.to_mapping()
    assert (
        mapping["lengths_sha256"]
        == sha256(numpy.asarray([3, 1, 5], dtype="<i4").tobytes()).hexdigest()
    )
    assert {key: mapping[key] for key in ("shortest", "gap", "dense")} == {
        "shortest": 1,
        "gap": 2,
        "dense": False,
    }


@pytest.mark.parametrize(
    ("lengths", "gap", "error", "message"),
    [
        ((), 0, MetalFusedError, "1 to 65536 rows"),
        ([4], 0, MetalFusedError, "1 to 65536 rows"),
        ((0,), 0, MetalFusedError, "row length"),
        ((16385,), 0, MetalFusedError, "row length"),
        ((4,), 1025, MetalFusedError, "gap"),
        ((True,), 0, TypeError, "integer"),
        ((16384,) * 1025, 0, MetalFusedError, "exceeds"),
    ],
)
def test_a_layout_refuses_rows_it_cannot_launch(lengths, gap, error, message):
    with pytest.raises(error, match=message):
        RowLayout(lengths, gap)
    assert MAX_ELEMENTS == 1 << 24


def test_inputs_rebuild_from_the_seed_and_fill_every_gap():
    layout = RowLayout((5, 33, 1), CORRECTNESS_GAP)
    first = softmax_inputs(79, CORRECTNESS_STREAM, 2, layout, NORMAL)

    assert first.dtype == numpy.float32 and first.shape == (layout.buffer_elements,)
    assert numpy.array_equal(
        first, softmax_inputs(79, CORRECTNESS_STREAM, 2, layout, NORMAL)
    )
    gaps = numpy.ones(first.size, dtype=bool)
    gaps[layout.valid()] = False
    assert numpy.all(first[gaps] == GAP_INPUT)
    other = softmax_inputs(79, BENCHMARK_STREAM, 2, layout, NORMAL)
    assert not numpy.array_equal(first, other)
    packed = softmax_inputs(79, CORRECTNESS_STREAM, 2, RowLayout((5, 33, 1), 0), NORMAL)
    assert numpy.array_equal(packed, first[layout.valid()])


def test_each_named_distribution_stays_in_its_range():
    layout = dense_layout(8, 512)
    ranges = {
        NORMAL: (-7.0, 7.0),
        UNIFORM_80: (-80.0, 80.0),
        OFFSET_PLUS_1000: (993.0, 1007.0),
        OFFSET_MINUS_1000: (-1007.0, -993.0),
    }

    assert set(ranges) == set(DISTRIBUTIONS)
    for name, (low, high) in ranges.items():
        values = softmax_inputs(1, CORRECTNESS_STREAM, 0, layout, name)
        assert low <= values.min() and values.max() <= high, name
    with pytest.raises(MetalFusedError, match="distribution"):
        softmax_inputs(1, CORRECTNESS_STREAM, 0, layout, "cauchy")
    with pytest.raises(MetalFusedError, match="stream"):
        softmax_inputs(1, 7, 0, layout, NORMAL)


def test_a_ragged_layout_draws_its_lengths_from_its_own_stream():
    layout = ragged_layout(79, 4, 300, 4096)

    assert layout == ragged_layout(79, 4, 300, 4096)
    assert layout != ragged_layout(79, 5, 300, 4096)
    assert layout.gap == 0 and layout.rows == 300
    assert 1 <= min(layout.lengths) and max(layout.lengths) <= 4096
    assert len(set(layout.lengths)) > 200


def test_the_reference_is_the_float64_softmax_of_each_row():
    layout = RowLayout((1, 31, 1000), CORRECTNESS_GAP)
    buffer = softmax_inputs(79, CORRECTNESS_STREAM, 1, layout, UNIFORM_80)

    reference = softmax_reference(buffer, layout)

    start = 0
    for row_start, length in zip(layout.starts(), layout.lengths):
        row = buffer[row_start : row_start + length].astype(numpy.float64)
        top = max(row)
        terms = [math.exp(value - top) for value in row]
        total = math.fsum(terms)
        expected = numpy.asarray([term / total for term in terms])
        got = reference.output[start : start + length]
        assert numpy.allclose(got, expected, rtol=1e-14, atol=1e-300)
        assert reference.distance[start : start + length].min() == 0.0
        start += length
    assert reference.output[0] == 1.0


def test_the_reference_refuses_a_buffer_that_is_not_its_layout():
    layout = dense_layout(2, 3)
    with pytest.raises(MetalFusedError, match="layout's size"):
        softmax_reference(numpy.zeros(5, dtype=numpy.float32), layout)
    with pytest.raises(MetalFusedError, match="layout's size"):
        softmax_reference(numpy.zeros(6, dtype=numpy.float64), layout)
    with pytest.raises(MetalFusedError, match="finite"):
        softmax_reference(numpy.full(6, numpy.inf, dtype=numpy.float32), layout)


@pytest.mark.parametrize(
    ("variant", "length", "size", "expected"),
    [
        (U0, 1000, 64, (21, 1)),
        (F1, 1, 32, (5, 1)),
        (F2, 33, 32, (32, 1)),
        (F2, 16384, 1024, (77, 1)),
        (F3, 4096, 256, (55, 18)),
        (F3, 1, 32, (33, 3)),
    ],
)
def test_each_order_counts_the_roundings_and_exp_calls_of_a_term(
    variant, length, size, expected
):
    assert order_terms(variant, length, size) == expected


def test_a_bound_holds_the_flush_floor_and_widens_with_the_order():
    layout = dense_layout(2, 4096)
    buffer = softmax_inputs(79, CORRECTNESS_STREAM, 0, layout, NORMAL)
    reference = softmax_reference(buffer, layout)

    widths = {}
    for variant in (U0, F1, F2, F3):
        bounds = error_bounds(variant, reference)
        assert numpy.all(bounds.elements >= FLOOR)
        assert bounds.row_sums.shape == (2,)
        widths[variant] = float(numpy.max(bounds.elements / reference.output))
    assert widths[U0] == widths[F1] < widths[F2] < widths[F3] < 2.0**-16
    assert widths[U0] > EXP_ERROR
    with pytest.raises(MetalFusedError, match="registered step"):
        error_bounds("torch_softmax", reference)


def test_row_sums_are_measured_in_float64_from_the_packed_outputs():
    layout = RowLayout((2, 3), 0)
    output = numpy.asarray([0.25, 0.75, 0.5, 0.25, 0.25], dtype=numpy.float32)

    assert row_sum_errors(output, layout).tolist() == [0.0, 0.0]
    shifted = output.copy()
    shifted[0] = numpy.float32(0.5)
    assert row_sum_errors(shifted, layout).tolist() == [0.25, 0.0]
    with pytest.raises(MetalFusedError, match="packed float32"):
        row_sum_errors(output[:4], layout)


def test_only_the_offset_inputs_break_softmax_without_the_max_subtraction():
    layout = dense_layout(2, 4096)
    finite = {
        name: naive_float32_finite(softmax_inputs(79, 0, 0, layout, name), layout)
        for name in DISTRIBUTIONS
    }

    assert finite == {
        NORMAL: True,
        UNIFORM_80: True,
        OFFSET_PLUS_1000: False,
        OFFSET_MINUS_1000: False,
    }


def test_a_sentinel_buffer_holds_the_fixed_quiet_nan():
    buffer = sentinel_buffer(RowLayout((3,), 2))

    assert buffer.dtype == numpy.float32 and buffer.size == 5
    assert set(buffer.view(numpy.uint32).tolist()) == {SENTINEL_BITS}


def test_the_exp_probe_grid_and_its_error_rule():
    arguments = exp_probe_arguments()
    exact = numpy.exp(arguments.astype(numpy.float64))

    assert arguments.dtype == numpy.float32 and arguments.size == EXP_PROBE_COUNT
    assert arguments[0] == -104.0 and arguments[-9] == 0.0
    assert arguments[-8:].tolist() == [
        float(numpy.float32(value)) for value in EXP_PROBE_EXTREMES
    ]
    close = exp_probe_errors(exact.astype(numpy.float32))
    assert close.holds and close.max_ulps <= 0.5
    assert close.tiny_results + close.normal_results == EXP_PROBE_COUNT
    off = exact.astype(numpy.float32)
    off[-1000] = numpy.float32(off[-1000] * (1 + 2.0**-20))
    assert not exp_probe_errors(off).holds
    flushed = exact.astype(numpy.float32)
    flushed[exact < 2.0**-126] = 0.0
    assert exp_probe_errors(flushed).holds
    with pytest.raises(MetalFusedError, match="one float32 per argument"):
        exp_probe_errors(exact)
    broken = exact.astype(numpy.float32)
    broken[5] = numpy.nan
    with pytest.raises(MetalFusedError, match="not finite"):
        exp_probe_errors(broken)


def test_hashes_cover_the_little_endian_bytes():
    values = numpy.asarray([1.0, -2.5], dtype=numpy.float32)

    assert float32_sha256(values) == sha256(values.astype("<f4").tobytes()).hexdigest()
    assert float64_sha256(values.astype(numpy.float64)) == (
        sha256(values.astype("<f8").tobytes()).hexdigest()
    )
    with pytest.raises(MetalFusedError):
        float32_sha256(values.astype(numpy.float64))
