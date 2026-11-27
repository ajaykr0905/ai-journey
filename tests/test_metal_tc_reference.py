import numpy
import pytest
import torch

from ai_journey.matmul_metal_reference import gamma, matmul_inputs
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.metal_tc_reference import (
    GUARD_BITS,
    INPUT_UNIT,
    SENTINEL_BITS,
    TAIL,
    bfloat16_round,
    bound_terms,
    guarded,
    input_sha256,
    operands,
    rounded,
    sentinel_output,
    tc_reference,
)
from ai_journey.metal_tc_source import BFLOAT16, FLOAT16, FLOAT32, MetalTcError

SHAPE = MatmulShape(37, 29, 300)


def test_bfloat_rounding_matches_the_torch_conversion_with_ties_to_even():
    generator = numpy.random.default_rng(5)
    values = generator.standard_normal(4096, dtype=numpy.float32)
    ties = numpy.array([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8), 2**-130], "float32")
    values = numpy.concatenate((values, ties))
    expected = torch.from_numpy(values).to(torch.bfloat16).float().numpy()

    assert numpy.array_equal(bfloat16_round(values), expected)
    assert bfloat16_round(ties[:3]).tolist() == [1.0, 1 + 2**-6, -1.0]
    with pytest.raises(MetalTcError):
        bfloat16_round(numpy.array([numpy.inf], dtype=numpy.float32))


def test_each_type_stores_its_rounded_values_exactly():
    values = operands(82, 0, 0, SHAPE).a
    half = rounded(values, FLOAT16)
    brain = rounded(values, BFLOAT16)

    assert rounded(values, FLOAT32) is values
    assert numpy.array_equal(half, values.astype(numpy.float16).astype("float32"))
    assert numpy.array_equal(torch.from_numpy(brain).to(torch.bfloat16).float(), brain)
    assert numpy.all(brain.view(numpy.uint32) & 0xFFFF == 0)
    for precision, unit in INPUT_UNIT.items():
        error = numpy.abs(rounded(values, precision) - values)
        assert numpy.all(error <= unit * numpy.abs(values) + 2.0**-14)
    with pytest.raises(MetalTcError):
        rounded(values, "float64")


def test_operands_come_from_the_day_74_stream_and_hash_in_their_width():
    first, again = operands(82, 1, 3, SHAPE), operands(82, 1, 3, SHAPE)
    left, right = matmul_inputs(82, 1, 3, SHAPE)

    assert numpy.array_equal(first.a, left) and numpy.array_equal(first.b, right)
    assert first.digests() == again.digests()
    assert len({digests["a_sha256"] for digests in first.digests().values()}) == 3
    assert input_sha256(first.a, FLOAT16) != input_sha256(first.a, BFLOAT16)
    with pytest.raises(MetalTcError):
        operands(82, 2, 0, SHAPE)


def test_the_bound_terms_follow_the_unit_roundoff_of_each_type_and_k():
    k = 1000
    g, r = gamma(k, 2.0**-24), gamma(k, 2.0**-53)

    assert bound_terms(FLOAT32, k)[0] == g + r
    half = bound_terms(FLOAT16, k)
    assert half[0] == g * (1 + 2**-11) ** 2 + 2 * 2**-11 + 2**-22 + r
    assert half[1] == 2.0**-14 * (1 + 2**-11) * (1 + g)
    assert bound_terms(BFLOAT16, k)[0] > half[0] > bound_terms(FLOAT32, k)[0]
    assert bound_terms(FLOAT32, 2 * k)[0] > bound_terms(FLOAT32, k)[0]


@pytest.mark.parametrize("precision", [FLOAT32, FLOAT16, BFLOAT16])
def test_a_float32_accumulation_of_each_type_stays_inside_its_bound(precision):
    values = operands(82, 0, 1, SHAPE)
    reference = tc_reference(values)
    a, b = values.inputs(precision)
    product = numpy.zeros((SHAPE.m, SHAPE.n), dtype=numpy.float32)
    for step in range(SHAPE.k):
        product = product + numpy.multiply.outer(a[:, step], b[step, :])

    error = numpy.abs(product.astype(numpy.float64) - reference.output)
    assert numpy.max(error / reference.bound(precision)) <= 1.0
    assert numpy.max(error / reference.vendor_bound(precision)) <= 1.0
    if precision != FLOAT32:
        assert numpy.max(error / reference.bound(FLOAT32)) > 1.0


def test_the_reference_adds_in_k_order_and_checks_rows_spread_evenly():
    values = operands(82, 1, 0, SHAPE)
    full = tc_reference(values)
    part = tc_reference(values, 5)
    exact = values.a.astype(numpy.float64) @ values.b.astype(numpy.float64)

    assert full.rows.tolist() == list(range(SHAPE.m))
    assert part.rows.tolist() == [0, 9, 18, 27, 36]
    assert numpy.allclose(full.output, exact, rtol=0, atol=1e-9)
    assert numpy.array_equal(part.output, full.output[part.rows])
    assert numpy.array_equal(part.column_sums, full.column_sums)
    assert part.row_sums.tolist() == full.row_sums[part.rows].tolist()
    assert full.digests() == tc_reference(values).digests()


def test_guards_and_sentinels_have_fixed_bits():
    values = operands(82, 0, 2, SHAPE)
    buffer = guarded(values.b)
    output = sentinel_output(SHAPE)

    assert buffer.size == SHAPE.k * SHAPE.n + TAIL
    assert numpy.all(buffer[-TAIL:].view(numpy.uint32) == GUARD_BITS)
    assert output.size == SHAPE.m * SHAPE.n + TAIL
    assert numpy.all(output.view(numpy.uint32) == SENTINEL_BITS)
