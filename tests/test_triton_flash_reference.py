"""Tests of the blockwise float32 reference, the float64 naive form and the bound."""

import math

import numpy
import pytest

from ai_journey.triton_flash_kernel import TritonFlashError, float32_scale
from ai_journey.triton_flash_reference import (
    FAMILIES,
    LARGE,
    NORMAL,
    AttentionCase,
    attention_bounds,
    attention_inputs,
    blockwise_attention,
    naive_attention,
    naive_float32,
    unshifted_overflows,
    visible,
)

CASES = (
    AttentionCase(1, 1, 16, False),
    AttentionCase(2, 17, 16, True),
    AttentionCase(2, 70, 32, False),
    AttentionCase(1, 129, 64, True),
    AttentionCase(1, 40, 128, True),
)
BLOCKS = ((16, 16), (64, 32), (128, 64))


def ratios(case, family, block_m, block_n):
    q, k, v = attention_inputs(family, 74, 0, case)
    output, maxima, sums = blockwise_attention(q, k, v, case.causal, block_m, block_n)
    reference, lse = naive_attention(q, k, v, case.causal)
    bounds = attention_bounds(q, k, v, case.causal, block_n)
    out, log = [], []
    for head, bound in enumerate(bounds):
        error = numpy.abs(output[head] - reference[head]).max()
        rebuilt = maxima[head] + numpy.log(sums[head].astype(numpy.float64))
        out.append(error / bound.output)
        log.append(numpy.abs(rebuilt - lse[head]).max() / bound.lse)
    return max(out), max(log), sums


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("case", CASES, ids=lambda case: f"n{case.sequence}")
@pytest.mark.parametrize("blocks", BLOCKS, ids=lambda pair: f"m{pair[0]}n{pair[1]}")
def test_the_blockwise_loop_meets_the_float64_reference_within_the_bound(
    family, case, blocks
):
    output_ratio, lse_ratio, sums = ratios(case, family, *blocks)

    assert output_ratio <= 1
    assert lse_ratio <= 1
    # The row max adds exp(0) = 1, and a sum of non-negative floats never falls.
    assert (sums >= 1).all()


def test_the_bound_is_loose_but_not_vacuous():
    # The recorded worst ratio stays far below 1, and the bound stays below
    # the largest value, so a zero or random output fails it.
    output_ratio, lse_ratio, _ = ratios(CASES[3], LARGE, 64, 32)
    q, k, v = attention_inputs(LARGE, 74, 0, CASES[3])
    bound = attention_bounds(q, k, v, True, 32)[0]

    assert 0 < output_ratio < 0.05 and 0 < lse_ratio < 0.05
    assert bound.output < 0.1 * numpy.abs(v).max()


def test_large_scores_overflow_without_the_shift_but_not_in_the_online_softmax():
    case = CASES[2]
    q, k, v = attention_inputs(LARGE, 74, 0, case)
    output, _, _ = blockwise_attention(q, k, v, case.causal, 16, 16)

    assert unshifted_overflows(q, k, case.causal) > 0
    assert numpy.isfinite(output).all()
    normal = attention_inputs(NORMAL, 74, 0, case)
    assert unshifted_overflows(normal[0], normal[1], case.causal) == 0


def test_a_causal_row_ignores_later_keys():
    case = CASES[1]
    q, k, v = attention_inputs(NORMAL, 74, 0, case)
    changed = v.copy()
    changed[:, 9:] = 1000.0

    first, _, _ = blockwise_attention(q, k, v, True, 16, 16)
    second, _, _ = blockwise_attention(q, k, changed, True, 16, 16)
    full, _, _ = blockwise_attention(q, k, changed, False, 16, 16)

    assert numpy.array_equal(first[:, :9], second[:, :9])
    assert not numpy.allclose(first[:, :9], full[:, :9])
    assert visible(3, True).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_a_skipped_rescale_breaks_the_bound():
    # Without alpha, the old terms keep their old max: the sum is wrong.
    case = CASES[3]
    q, k, v = attention_inputs(LARGE, 74, 0, case)
    reference, _ = naive_attention(q, k, v, True)
    bound = attention_bounds(q, k, v, True, 16)[0].output
    output = numpy.zeros_like(q)
    scale = numpy.float32(float32_scale(64))
    s = (q[0] @ k[0].T) * scale
    s[~visible(129, True)] = -numpy.inf
    for row in range(129):
        acc = numpy.zeros(64, dtype=numpy.float32)
        total = numpy.float32(0)
        top = numpy.float32(-numpy.inf)
        for start in range(0, 129, 16):
            block = s[row, start : start + 16]
            top = max(top, block.max())
            weights = numpy.exp(block - top)
            total = total + weights.sum()
            acc = acc + weights @ v[0, start : start + 16]
        output[0, row] = acc / total

    assert numpy.abs(output[0] - reference[0]).max() > bound


def test_the_float32_naive_form_agrees_and_the_scale_is_float32():
    case = CASES[2]
    q, k, v = attention_inputs(NORMAL, 74, 0, case)
    reference, _ = naive_attention(q, k, v, case.causal)

    assert numpy.abs(naive_float32(q, k, v, case.causal) - reference).max() < 1e-5
    assert float32_scale(64) == 0.125
    assert float32_scale(32) == float(numpy.float32(1 / math.sqrt(32)))
    assert float32_scale(32) != 1 / math.sqrt(32)


def test_inputs_are_seeded_per_family_and_case():
    case = CASES[1]
    q, k, v = attention_inputs(NORMAL, 74, 1, case)
    again = attention_inputs(NORMAL, 74, 1, case)
    large = attention_inputs(LARGE, 74, 1, case)

    assert all(numpy.array_equal(x, y) for x, y in zip((q, k, v), again))
    assert q.dtype == numpy.float32 and q.shape == (2, 17, 16)
    assert not numpy.array_equal(q, attention_inputs(NORMAL, 74, 2, case)[0])
    assert numpy.abs(large[0]).max() > 8 * 2


@pytest.mark.parametrize(
    ("fields", "error"),
    (
        ((0, 4, 16, False), TritonFlashError),
        ((1, 0, 16, False), TritonFlashError),
        ((1, 1025, 16, False), TritonFlashError),
        ((1, 4, 24, False), TritonFlashError),
        ((1, 4, 16, 1), TypeError),
        ((True, 4, 16, False), TypeError),
    ),
)
def test_a_case_refuses_bad_fields(fields, error):
    with pytest.raises(error):
        AttentionCase(*fields)


def test_the_references_refuse_bad_operands():
    case = CASES[1]
    q, k, v = attention_inputs(NORMAL, 74, 0, case)

    with pytest.raises(TritonFlashError, match="float32 arrays"):
        naive_attention(q.astype(numpy.float64), k, v, True)
    with pytest.raises(TritonFlashError, match="float32 arrays"):
        blockwise_attention(q, k[:, :5], v, True, 16, 16)
    with pytest.raises(TritonFlashError, match="family"):
        attention_inputs("tiny", 74, 0, case)
