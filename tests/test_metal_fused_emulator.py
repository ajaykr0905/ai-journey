"""A NumPy float32 emulation of each softmax step, in the order of its kernel.

The emulation follows the slots of each thread, the tree of U0 and F1, one
order of ``simd_sum`` for F2 and F3, and the online update of F3, with
float32 rounding at each step and results below 2**-126 flushed to zero.
NumPy's float32 exp stands in for ``precise::exp``. The emulation is not the
GPU, so it shows that the declared bounds hold for real float32 rounding in
these orders, and it serves as the stand-in backend of the CPU tests.
"""

import numpy
import pytest

from ai_journey.metal_fused_reference import (
    CORRECTNESS_GAP,
    NORMAL,
    OFFSET_MINUS_1000,
    OFFSET_PLUS_1000,
    UNIFORM_80,
    RowLayout,
    dense_layout,
    error_bounds,
    row_sum_errors,
    sentinel_buffer,
    softmax_inputs,
    softmax_reference,
)
from ai_journey.metal_fused_source import (
    F1,
    F2,
    F3,
    LOWEST,
    SIMD_WIDTH,
    SLOTS,
    STEP_VARIANTS,
    U0,
)

F32 = numpy.float32
LOWEST32 = F32(LOWEST)
TINY = numpy.finfo(F32).tiny


def _exp(values):
    with numpy.errstate(under="ignore"):
        result = numpy.exp(numpy.asarray(values, dtype=F32))
    return numpy.where(numpy.abs(result) < TINY, F32(0), result).astype(F32)


def _fma(a, b, c):
    """float32 fma: the float64 product of two float32 values is exact."""

    return (a.astype(numpy.float64) * b.astype(numpy.float64) + c).astype(F32)


def _tree(values, combine):
    values = values.copy()
    stride = values.size // 2
    while stride > 0:
        values[:stride] = combine(values[:stride], values[stride : 2 * stride])
        stride //= 2
    return values[0]


def _in_order(values):
    total = F32(0)
    for value in values:
        total = F32(total + value)
    return total


def _slots(row, size):
    columns = numpy.arange(SLOTS)[:, None] * size + numpy.arange(size)[None, :]
    valid = columns < row.size
    values = numpy.full((SLOTS, size), LOWEST32, dtype=F32)
    values[valid] = row[columns[valid]]
    return values, valid, columns


def _thread_sums(terms):
    sums = numpy.zeros(terms.shape[1], dtype=F32)
    for slot in terms:
        sums = (sums + slot).astype(F32)
    return sums


def emulate_row(variant, row, size):
    """Return the float32 softmax of one row in the order of one step."""

    values, valid, columns = _slots(row, size)
    groups = size // SIMD_WIDTH
    if variant in (U0, F1, F2):
        thread_max = values.max(axis=0)
        if variant == F2:
            row_max = thread_max.reshape(groups, SIMD_WIDTH).max()
        else:
            row_max = _tree(thread_max, numpy.maximum)
        terms = numpy.where(valid, _exp((values - row_max).astype(F32)), F32(0))
        sums = _thread_sums(terms.astype(F32))
        if variant == F2:
            partials = [_in_order(part) for part in sums.reshape(groups, SIMD_WIDTH)]
            total = _in_order(partials)
        else:
            total = _tree(sums, numpy.add)
        outputs = (terms / total).astype(F32)
    elif variant == F3:
        running = numpy.full(size, LOWEST32, dtype=F32)
        sums = numpy.zeros(size, dtype=F32)
        for slot, ok in zip(values, valid):
            low = numpy.minimum(running, slot)
            high = numpy.maximum(running, slot)
            scale = _exp((low - high).astype(F32))
            grown = _fma(sums, scale, 1.0)
            kept = (sums + scale).astype(F32)
            sums = numpy.where(ok, numpy.where(slot > running, grown, kept), sums)
            running = numpy.where(ok, high, running)
        group_max, group_sum = [], []
        for part_max, part_sum in zip(
            running.reshape(groups, SIMD_WIDTH), sums.reshape(groups, SIMD_WIDTH)
        ):
            top = part_max.max()
            scaled = (part_sum * _exp((part_max - top).astype(F32))).astype(F32)
            group_max.append(top)
            group_sum.append(_in_order(scaled))
        group_max = numpy.asarray(group_max, dtype=F32)
        group_sum = numpy.asarray(group_sum, dtype=F32)
        row_max = group_max.max()
        weights = (group_sum * _exp((group_max - row_max).astype(F32))).astype(F32)
        total = _in_order(weights)
        outputs = (_exp((values - row_max).astype(F32)) / total).astype(F32)
    else:
        raise ValueError(variant)
    result = numpy.empty(row.size, dtype=F32)
    result[columns[valid]] = outputs[valid]
    return numpy.where(numpy.abs(result) < TINY, F32(0), result).astype(F32)


def emulate(variant, buffer, layout):
    """Return a whole output buffer: emulated rows, sentinels in every gap."""

    output = sentinel_buffer(layout)
    size = layout.group_size
    for start, length in zip(layout.starts(), layout.lengths):
        start = int(start)
        output[start : start + length] = emulate_row(
            variant, buffer[start : start + length], size
        )
    return output


LAYOUTS = (
    (dense_layout(4, 1, CORRECTNESS_GAP), NORMAL),
    (dense_layout(6, 31, CORRECTNESS_GAP), UNIFORM_80),
    (dense_layout(6, 33, CORRECTNESS_GAP), NORMAL),
    (dense_layout(3, 1000, CORRECTNESS_GAP), UNIFORM_80),
    (dense_layout(2, 4096, CORRECTNESS_GAP), NORMAL),
    (dense_layout(2, 4096, CORRECTNESS_GAP), OFFSET_PLUS_1000),
    (dense_layout(2, 1000, CORRECTNESS_GAP), OFFSET_MINUS_1000),
    (RowLayout((1, 31, 33, 1000, 4096, 16384, 2049, 7), CORRECTNESS_GAP), UNIFORM_80),
)


@pytest.mark.parametrize("variant", STEP_VARIANTS)
@pytest.mark.parametrize(("layout", "distribution"), LAYOUTS)
def test_each_emulated_order_meets_its_declared_bounds(variant, layout, distribution):
    buffer = softmax_inputs(79, 0, 3, layout, distribution)
    reference = softmax_reference(buffer, layout)
    bounds = error_bounds(variant, reference)

    output = emulate(variant, buffer, layout)
    packed = output[layout.valid()]
    gaps = numpy.ones(layout.buffer_elements, dtype=bool)
    gaps[layout.valid()] = False

    assert numpy.all(numpy.isfinite(packed))
    assert numpy.all(output[gaps].view(numpy.uint32) == 0x7FC00000)
    error = numpy.abs(packed.astype(numpy.float64) - reference.output)
    assert numpy.all(error <= bounds.elements)
    assert numpy.all(row_sum_errors(packed, layout) <= bounds.row_sums)


def test_u0_and_f1_run_the_same_float32_arithmetic():
    layout, distribution = LAYOUTS[-1]
    buffer = softmax_inputs(79, 0, 5, layout, distribution)

    u0 = emulate(U0, buffer, layout).view(numpy.uint32)
    f1 = emulate(F1, buffer, layout).view(numpy.uint32)

    assert numpy.array_equal(u0, f1)


def test_the_bounds_catch_a_relative_error_of_two_to_the_minus_sixteen():
    for variant in STEP_VARIANTS:
        for layout, distribution in LAYOUTS:
            buffer = softmax_inputs(79, 0, 7, layout, distribution)
            reference = softmax_reference(buffer, layout)
            bounds = error_bounds(variant, reference)
            wrong = reference.output * (1 + 2.0**-16)
            assert numpy.any(numpy.abs(wrong - reference.output) > bounds.elements), (
                variant,
                layout.lengths[:2],
                distribution,
            )
