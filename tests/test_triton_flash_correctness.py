"""Tests of the correctness harness with a NumPy stand-in for the interpreter."""

import base64

import numpy
import pytest

from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from ai_journey.triton_flash_config import PINNED_PLAN
from ai_journey.triton_flash_correctness import (
    check_case,
    encode_float32,
    sentinel_buffer,
)
from ai_journey.triton_flash_kernel import (
    FLASH_CONFIGS,
    TRITON_SOURCE,
    TritonFlashError,
    float32_scale,
    padded_rows,
)
from ai_journey.triton_flash_reference import (
    FAMILIES,
    AttentionCase,
    array_sha256,
    blockwise_attention,
    visible,
)
from test_triton_flash_emulated import EmulatedBackend, install

STAND_IN_ENVIRONMENT = {
    "python": "3.11.14",
    "implementation": "CPython",
    "system": "Darwin",
    "machine": "arm64",
    "numpy": numpy.__version__,
    "torch": "none",
    "triton": "none",
    "triton_interpret": "unset",
}


def _no_rescale(q, k, v, causal, block_n):
    """The online softmax without alpha: old terms keep their old max."""

    heads, n, dim = q.shape
    scale = numpy.float32(float32_scale(dim))
    output = numpy.zeros_like(q)
    maxima = numpy.zeros((heads, n), dtype=numpy.float32)
    sums = numpy.zeros((heads, n), dtype=numpy.float32)
    mask = visible(n, causal)
    for head in range(heads):
        s = (q[head] @ k[head].T) * scale
        s[~mask] = -numpy.inf
        top = numpy.full(n, -numpy.inf, dtype=numpy.float32)
        total = numpy.zeros(n, dtype=numpy.float32)
        acc = numpy.zeros((n, dim), dtype=numpy.float32)
        for start in range(0, n, block_n):
            block = s[:, start : start + block_n]
            top = numpy.maximum(top, block.max(axis=1))
            weights = numpy.exp(block - top[:, None])
            total = total + weights.sum(axis=1)
            acc = acc + weights @ v[head, start : start + block_n]
        output[head], maxima[head], sums[head] = acc / total[:, None], top, total
    return output, maxima, sums


class StandInBackend:
    """The blockwise reference in the buffers of a run, with one optional fault."""

    name = "injected"

    def __init__(self, fault=None):
        self.fault = fault
        self.environment = dict(STAND_IN_ENVIRONMENT)
        self.source_sha256 = None
        self.calls = 0

    def run(self, q, k, v, *, causal, config, output, maxima, sums):
        self.calls += 1
        heads, n, dim = q.shape
        rows = padded_rows(n, config)
        if self.fault == "skip_rescale":
            o, m, l_ = _no_rescale(q, k, v, causal, config.block_n)
        else:
            o, m, l_ = blockwise_attention(
                q, k, v, causal, config.block_m, config.block_n
            )
        out = output.reshape(heads, rows, dim)
        top = maxima.reshape(heads, rows)
        total = sums.reshape(heads, rows)
        stop = n - 1 if self.fault == "skip_last_row" else n
        out[:, :stop], top[:, :stop], total[:, :stop] = (
            o[:, :stop],
            m[:, :stop],
            l_[:, :stop],
        )
        if self.fault == "unmasked_store" and rows > n:
            out[:, n] = 0.0
        if self.fault == "short_buffer":
            return output[:-1], maxima, sums
        if self.fault == "two_buffers":
            return output, maxima
        return output, maxima, sums


CASE = AttentionCase(2, 17, 16, True)


def test_every_config_holds_both_families_on_every_pinned_case():
    backend = StandInBackend()
    for index, case in enumerate(PINNED_PLAN.cases):
        record = check_case(
            backend, seed=81, index=index, case=case, configs=FLASH_CONFIGS
        )

        assert len(record.runs) == len(FLASH_CONFIGS) * len(FAMILIES)
        assert [operand.family for operand in record.operands] == list(FAMILIES)
        for run in record.runs:
            assert 0 <= run.output_bound_ratio <= 1
            assert 0 <= run.lse_bound_ratio <= 1
            assert len(run.output_bounds) == case.batch_heads
    assert backend.calls == len(PINNED_PLAN.cases) * 8


def test_a_record_keeps_every_value_and_its_digest():
    record = check_case(
        StandInBackend(), seed=81, index=1, case=CASE, configs=(FLASH_CONFIGS[3],)
    )
    run = record.runs[0]
    values = numpy.frombuffer(base64.b64decode(run.output), dtype="<f4")

    assert values.size == 2 * 17 * 16
    assert array_sha256(values.reshape(2, 17, 16)) == run.output_sha256
    assert (run.programs, run.key_blocks, run.guard_rows) == (4, 2, 15)
    assert record.to_mapping()["scale"] == 0.25
    assert encode_float32(numpy.float32([1.0])) == "AACAPw=="


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("skip_rescale", "times its bound"),
        ("skip_last_row", "not finite"),
        ("unmasked_store", "escaped its mask"),
        ("short_buffer", "no float32 buffer"),
        ("two_buffers", "must return O, m and l"),
    ),
)
def test_each_fault_breaks_a_rule(fault, message):
    case = AttentionCase(1, 129, 64, True)

    with pytest.raises(TritonFlashError, match=message):
        check_case(
            StandInBackend(fault),
            seed=81,
            index=3,
            case=case,
            configs=(FLASH_CONFIGS[3],),
        )


def test_the_sentinel_buffer_holds_the_day_73_bits():
    buffer = sentinel_buffer(3)

    assert buffer.dtype == numpy.float32
    assert (buffer.view(numpy.uint32) == FLOAT32_SENTINEL_BITS).all()


def test_a_case_needs_a_tuple_of_configs():
    with pytest.raises(TritonFlashError, match="non-empty tuple"):
        check_case(StandInBackend(), seed=81, index=0, case=CASE, configs=[])


def test_the_pinned_text_passes_the_harness_under_the_language_model(monkeypatch):
    install(monkeypatch)
    backend = EmulatedBackend()

    for index, case in enumerate(PINNED_PLAN.cases):
        record = check_case(
            backend, seed=81, index=index, case=case, configs=FLASH_CONFIGS
        )
        assert all(run.output_bound_ratio <= 1 for run in record.runs)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "tl.store(o_ptrs, acc, mask=row_mask[:, None])",
            "tl.store(o_ptrs, acc)",
            "escaped its mask",
        ),
        (
            "alpha = tl.exp(m_i - m_new)",
            "alpha = tl.exp(m_new - m_new)",
            "times its bound",
        ),
        (
            "acc = acc / l_i[:, None]",
            "acc = acc / (l_i[:, None] + 1)",
            "times its bound",
        ),
    ),
)
def test_a_changed_text_fails_the_harness_under_the_language_model(
    monkeypatch, old, new, message
):
    install(monkeypatch)
    assert TRITON_SOURCE.count(old) == 1
    backend = EmulatedBackend(TRITON_SOURCE.replace(old, new))

    with pytest.raises(TritonFlashError, match=message):
        check_case(
            backend,
            seed=81,
            index=3,
            case=AttentionCase(1, 129, 64, True),
            configs=(FLASH_CONFIGS[3],),
        )
