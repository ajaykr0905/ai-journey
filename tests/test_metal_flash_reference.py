import numpy
import pytest

from ai_journey.metal_flash_reference import (
    CORRECTNESS_GAP,
    FLOAT32_LOG_MAX,
    MAX_SCORE_ELEMENTS,
    FlashLayout,
    check_rows,
    dense_layout,
    flash_inputs,
    flash_reference,
    gather_rows,
    sentinel_output,
    unscaled_exp_overflows,
)
from ai_journey.metal_flash_source import CAUSAL, FULL, MODES, MetalFlashError
from ai_journey.metal_fused_reference import GAP_INPUT, SENTINEL_BITS

RAGGED = FlashLayout((1, 31, 33, 100), CORRECTNESS_GAP)


def float32_attention(q, k, v, causal):
    scores = (q @ k.T) * numpy.float32(0.125)
    if causal:
        scores = numpy.where(numpy.tril(numpy.ones(scores.shape, bool)), scores, -1e30)
    weights = numpy.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ v


def test_the_layout_places_each_sequence_and_each_group():
    layout = RAGGED

    assert layout.starts().tolist() == [0, 4, 38, 74]
    assert layout.buffer_rows == 165 + 4 * 3
    assert layout.score_elements == 1 + 31**2 + 33**2 + 100**2
    assert layout.blocks() == 1 + 1 + 2 + 4
    seqs, rows = layout.query_groups()
    assert seqs.size == rows.size == 165
    assert (int(seqs[40]), int(rows[40])) == (2, 8)
    seqs, blocks = layout.block_groups()
    assert seqs.tolist() == [0, 1, 2, 2, 3, 3, 3, 3]
    assert blocks.tolist() == [0, 0, 0, 1, 0, 1, 2, 3]
    assert layout.score_starts().tolist() == [0, 1, 962, 2051]
    assert layout.pairs(FULL) == layout.score_elements
    assert layout.pairs(CAUSAL) == 1 + 31 * 32 // 2 + 33 * 34 // 2 + 100 * 101 // 2
    assert layout.tile_loads(FULL) == 1 + 1 + 2 * 2 + 4 * 4
    assert layout.tile_loads(CAUSAL) == 1 + 1 + (1 + 2) + (1 + 2 + 3 + 4)


def test_a_layout_beyond_its_bounds_is_refused():
    with pytest.raises(MetalFlashError, match="sequence length"):
        FlashLayout((8193,), 0)
    with pytest.raises(MetalFlashError, match="scores"):
        dense_layout(5, 8192)
    assert dense_layout(4, 8192).score_elements == MAX_SCORE_ELEMENTS
    with pytest.raises(MetalFlashError, match="sequences"):
        FlashLayout((), 0)


def test_the_inputs_are_seeded_and_every_gap_row_holds_the_gap_input():
    one = flash_inputs(81, 0, 2, RAGGED, "wide_scores")
    two = flash_inputs(81, 0, 2, RAGGED, "wide_scores")
    other = flash_inputs(81, 1, 2, RAGGED, "wide_scores")

    assert one.digests() == two.digests() != other.digests()
    gaps = numpy.ones(RAGGED.buffer_rows, dtype=bool)
    gaps[RAGGED.valid_rows()] = False
    for array in (one.q, one.k, one.v):
        assert numpy.all(array.reshape(-1, 64)[gaps] == numpy.float32(GAP_INPUT))
    q, _, v = flash_inputs(81, 0, 0, RAGGED, "offset_values").sequence(3)
    assert q.shape == v.shape == (100, 64)
    assert 990 < float(v.mean()) < 1010
    assert sentinel_output(RAGGED).view(numpy.uint32)[0] == SENTINEL_BITS


def test_numpy_adds_the_middle_axis_one_key_at_a_time():
    """The reference digests assume this order on every host."""

    values = numpy.random.default_rng(1).standard_normal((3, 257, 129))
    total = numpy.zeros((3, 129))
    for key in range(257):
        total = total + values[:, key, :]

    assert numpy.array_equal(values.sum(axis=1), total)


@pytest.mark.parametrize("mode", MODES)
def test_the_reference_matches_float64_attention_and_bounds_float32(mode):
    inputs = flash_inputs(81, 0, 0, RAGGED, "normal")
    reference = flash_reference(inputs, mode)

    assert reference.rows.tolist() == RAGGED.valid_rows().tolist()
    first = 0
    for index, length in enumerate(RAGGED.lengths):
        q, k, v = inputs.sequence(index)
        exact = float32_attention(
            *(a.astype(numpy.float64) for a in (q, k, v)), mode == CAUSAL
        )
        rows = slice(first, first + length)
        assert numpy.allclose(reference.output[rows], exact, rtol=1e-12, atol=1e-12)
        single = float32_attention(q, k, v, mode == CAUSAL)
        error = numpy.abs(single - reference.output[rows])
        assert numpy.all(error <= reference.bound[rows])
        assert numpy.all(error <= reference.vendor_bound[rows])
        first += length
    assert numpy.all(reference.bound > 0)


def test_a_causal_first_query_sees_only_its_own_key():
    inputs = flash_inputs(81, 0, 0, dense_layout(1, 40), "normal")
    reference = flash_reference(inputs, CAUSAL)
    _, _, v = inputs.sequence(0)

    assert numpy.array_equal(reference.output[0], v[0].astype(numpy.float64))


def test_wide_scores_overflow_an_unscaled_float32_exp():
    wide = flash_reference(flash_inputs(81, 0, 1, RAGGED, "wide_scores"), FULL)
    normal = flash_reference(flash_inputs(81, 0, 1, RAGGED, "normal"), FULL)

    assert wide.max_score >= FLOAT32_LOG_MAX and unscaled_exp_overflows(wide)
    assert not unscaled_exp_overflows(normal)


def test_a_count_checks_both_ends_of_each_sequence():
    assert check_rows(10, None).tolist() == list(range(10))
    assert check_rows(4, 8).tolist() == [0, 1, 2, 3]
    rows = check_rows(1000, 5).tolist()
    assert rows[0] == 0 and rows[-1] == 999 and len(rows) == 5
    layout = dense_layout(2, 100)
    reference = flash_reference(flash_inputs(81, 1, 0, layout, "normal"), FULL, 4)
    assert reference.rows.tolist() == [0, 33, 66, 99, 100, 133, 166, 199]
    buffer = numpy.arange(layout.buffer_elements, dtype=numpy.float32)
    assert gather_rows(buffer, reference.rows).shape == (8, 64)


def test_the_reference_does_not_call_the_libm_exp(monkeypatch):
    inputs = flash_inputs(81, 0, 0, RAGGED, "normal")
    before = flash_reference(inputs, FULL)

    def refuse(*args, **kwargs):
        raise AssertionError("the reference must not call numpy.exp")

    monkeypatch.setattr(numpy, "exp", refuse)
    after = flash_reference(inputs, FULL)
    assert numpy.array_equal(before.output, after.output)
    assert numpy.array_equal(before.bound, after.bound)


def test_inputs_that_are_not_finite_or_unknown_are_refused():
    inputs = flash_inputs(81, 0, 0, dense_layout(1, 4), "normal")
    inputs.q[0] = numpy.inf
    with pytest.raises(MetalFlashError, match="finite"):
        flash_reference(inputs, FULL)
    with pytest.raises(MetalFlashError, match="distribution"):
        flash_inputs(81, 0, 0, RAGGED, "uniform")
    with pytest.raises(MetalFlashError, match="stream"):
        flash_inputs(81, 2, 0, RAGGED, "normal")
