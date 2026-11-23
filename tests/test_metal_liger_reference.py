import math

import numpy
import pytest

from ai_journey.metal_fused_reference import GAP_INPUT, SENTINEL_BITS
from ai_journey.metal_liger_reference import (
    CHUNK_ELEMENTS,
    LOG_ERROR,
    RSQRT_ERROR,
    LigerLayout,
    dense_layout,
    liger_inputs,
    liger_reference,
    log64,
    log_probe_arguments,
    math_probe_errors,
    naive_cross_entropy_finite,
    rsqrt_probe_arguments,
    sentinel_output,
    silu64,
    uint32_sha256,
)
from ai_journey.metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
)

RAGGED = LigerLayout((1, 31, 33, 1000, 4096, 7), 32)
F32 = numpy.float32


def float32_kernel(inputs):
    """A plain float32 NumPy version of each kernel, in another order of operations."""

    layout = inputs.layout
    valid = layout.valid()
    starts = layout.packed_starts()
    if inputs.family == SWIGLU:
        a, b = inputs.main[valid], inputs.second[valid]
        with numpy.errstate(over="ignore"):
            return (a / (F32(1) + numpy.exp(-a))) * b
    x = inputs.main[valid]
    rows = []
    for row, (start, length) in enumerate(zip(starts, layout.lengths)):
        values = x[start : start + length]
        if inputs.family == RMSNORM:
            mean = F32(numpy.sum(values * values, dtype=F32)) / F32(length)
            inverse = F32(1) / numpy.sqrt(mean + F32(1e-6))
            rows.append((values * inverse) * inputs.second[:length])
        else:
            top = values.max()
            total = numpy.sum(numpy.exp(values - top), dtype=F32)
            margin = top - values[inputs.second[row]]
            rows.append(numpy.array([margin + numpy.log(total)], dtype=F32))
    return numpy.concatenate(rows)


def test_log64_is_within_two_ulps_of_libm_and_exact_at_powers_of_two():
    values = numpy.concatenate(
        (
            numpy.geomspace(1e-300, 1e300, 20001),
            1.0 + numpy.arange(1, 2000) * 2.0**-40,
            numpy.linspace(1.0, 2.0**17, 5001),
        )
    )
    error = numpy.abs(log64(values) - numpy.log(values))

    assert numpy.all(error <= 2 * numpy.spacing(numpy.abs(numpy.log(values))) + 1e-300)
    assert log64(numpy.array([1.0]))[0] == 0.0
    assert log64(numpy.array([2.0**10]))[0] == 10 * math.log(2.0)
    with pytest.raises(MetalLigerError, match="positive finite"):
        log64(numpy.array([0.0]))
    with pytest.raises(MetalLigerError, match="positive finite"):
        log64(numpy.array([numpy.inf]))


def test_silu64_is_stable_at_both_signs():
    a = numpy.array([-1000.0, -80.0, -1.0, 0.0, 1.0, 80.0, 1000.0])
    expected = a / (1.0 + numpy.exp(-numpy.clip(a, -700, 700)))

    assert numpy.allclose(silu64(a)[1:], expected[1:], rtol=1e-15, atol=0)
    assert silu64(a)[0] == 0.0 and silu64(a)[-1] == 1000.0


def test_a_gapped_layout_places_rows_after_guards_and_chunks_long_batches():
    assert RAGGED.starts().tolist() == [0, 33, 96, 161, 1193, 5321]
    assert RAGGED.buffer_elements == RAGGED.elements + 6 * 32
    valid = RAGGED.valid()
    assert valid[:3].tolist() == [0, 33, 34] and valid.size == RAGGED.elements
    assert RAGGED.output_elements(CROSS_ENTROPY) == 6 + 32
    assert RAGGED.output_valid(CROSS_ENTROPY).tolist() == list(range(6))
    assert RAGGED.group_size(RMSNORM) == 256
    big = dense_layout(1024, 50257)
    chunks = list(big.chunks())
    assert chunks[0][0] == 0 and chunks[-1][1] == 1024
    assert all(
        (last - first) * 50257 <= CHUNK_ELEMENTS or last - first == 1
        for first, last in chunks
    )
    assert all(one[1] == two[0] for one, two in zip(chunks, chunks[1:]))
    with pytest.raises(MetalLigerError, match="exceeds"):
        dense_layout(1 << 16, 1 << 11)
    with pytest.raises(MetalLigerError, match="row length"):
        LigerLayout((0,), 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_inputs_rebuild_from_the_seed_with_guard_values_in_every_gap(family):
    one = liger_inputs(80, family, 0, 3, RAGGED, "uniform_80")
    two = liger_inputs(80, family, 0, 3, RAGGED, "uniform_80")
    other = liger_inputs(80, family, 1, 3, RAGGED, "uniform_80")

    assert one.digests() == two.digests() != other.digests()
    gaps = numpy.ones(RAGGED.buffer_elements, dtype=bool)
    gaps[RAGGED.valid()] = False
    assert numpy.all(one.main[gaps] == F32(GAP_INPUT))
    if family == RMSNORM:
        assert one.second.shape == (RAGGED.longest,)
    elif family == SWIGLU:
        assert numpy.all(one.second[gaps] == F32(GAP_INPUT))
    else:
        assert one.second.dtype == numpy.uint32
        assert numpy.all(one.second < RAGGED.lengths_array())
    sentinel = sentinel_output(family, RAGGED)
    assert numpy.all(sentinel.view(numpy.uint32) == SENTINEL_BITS)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "distribution", ["normal", "uniform_80", "offset_plus_1000", "offset_minus_1000"]
)
def test_a_float32_kernel_in_another_order_stays_inside_the_declared_bound(
    family, distribution
):
    inputs = liger_inputs(80, family, 0, 1, RAGGED, distribution)
    reference = liger_reference(inputs)
    error = numpy.abs(float32_kernel(inputs).astype(numpy.float64) - reference.output)

    assert numpy.all(numpy.isfinite(reference.output))
    assert numpy.max(error / reference.bound) <= 0.5
    assert numpy.max(error / reference.vendor_bound) <= 0.5
    assert numpy.all(reference.bound > 0)


def test_the_cross_entropy_bound_does_not_grow_with_the_logit_offset():
    layout = dense_layout(4, 1000)
    plain = liger_reference(liger_inputs(80, CROSS_ENTROPY, 0, 2, layout, "normal"))
    shifted = liger_reference(
        liger_inputs(80, CROSS_ENTROPY, 0, 2, layout, "offset_plus_1000")
    )

    assert numpy.max(shifted.bound) < 4 * numpy.max(plain.bound)
    assert numpy.max(shifted.bound) < 1e-4


def test_the_naive_float32_cross_entropy_fails_only_for_large_logits():
    layout = dense_layout(2, 64)

    assert naive_cross_entropy_finite(
        liger_inputs(80, CROSS_ENTROPY, 0, 0, layout, "normal")
    )
    assert not naive_cross_entropy_finite(
        liger_inputs(80, CROSS_ENTROPY, 0, 0, layout, "offset_plus_1000")
    )
    with pytest.raises(MetalLigerError, match="belongs to cross-entropy"):
        naive_cross_entropy_finite(liger_inputs(80, SWIGLU, 0, 0, layout, "normal"))


def test_the_math_probes_measure_the_assumed_errors():
    rsqrt = rsqrt_probe_arguments()
    exact = (1.0 / numpy.sqrt(rsqrt.astype(numpy.float64))).astype(F32)
    probe = math_probe_errors("rsqrt", exact)
    assert probe.holds and probe.max_ulps <= 0.5 and probe.exact_zero is None
    assert probe.to_mapping()["assumed_error"] == RSQRT_ERROR
    worse = exact * F32(1 + 2.0**-20)
    assert not math_probe_errors("rsqrt", worse).holds

    log = log_probe_arguments()
    exact = log64(log.astype(numpy.float64)).astype(F32)
    probe = math_probe_errors("log", exact)
    assert probe.holds and probe.exact_zero is True
    assert probe.to_mapping()["assumed_error"] == LOG_ERROR
    nonzero = exact.copy()
    nonzero[log == 1] = F32(2.0**-30)
    assert not math_probe_errors("log", nonzero).holds
    with pytest.raises(MetalLigerError, match="one float32 per argument"):
        math_probe_errors("log", exact[:-1])
    with pytest.raises(MetalLigerError, match="no probe"):
        math_probe_errors("exp", exact)


def test_uint32_arrays_hash_by_their_little_endian_bytes():
    values = numpy.array([1, 2, 3], dtype=numpy.uint32)

    assert uint32_sha256(values) == uint32_sha256(values.astype(">u4").astype("<u4"))
    with pytest.raises(MetalLigerError, match="uint32"):
        uint32_sha256(values.astype(numpy.int64))
