"""Run the pinned Triton text under a small NumPy model of ``triton.language``.

Triton has no macOS wheel, so these tests execute the exact text with fake
``triton`` and ``triton.language`` modules. The model covers only the calls
that the text makes: a pointer is a flat array and an offset array, a masked
load reads only the lanes of its mask, and ``tl.dot`` is ``numpy.matmul`` plus
the accumulator. The model is not the Triton interpreter. It checks the
indexing, the masks, the grid and the online softmax of the text against the
blockwise reference bit for bit. The interpreter tests check the real thing on
Linux.
"""

import math
import sys
import types

import numpy
import pytest

from ai_journey.triton_flash_kernel import (
    FLASH_CONFIGS,
    KERNEL_NAME,
    LAUNCHER_NAME,
    TRITON_SOURCE,
    float32_scale,
    padded_rows,
)
from ai_journey.triton_flash_reference import (
    FAMILIES,
    AttentionCase,
    attention_inputs,
    blockwise_attention,
)

SENTINEL = numpy.uint32(0x7FC00000)


class Pointer:
    """A flat float32 array and an offset, scalar or array, into it."""

    def __init__(self, base, offset=0):
        self.base, self.offset = base, offset

    def __add__(self, other):
        return Pointer(self.base, self.offset + numpy.asarray(other))

    __radd__ = __add__


class Language(types.ModuleType):
    """The parts of ``triton.language`` that the pinned text calls."""

    float32 = numpy.float32
    constexpr = object

    def __init__(self):
        super().__init__("triton.language")
        self.pid = (0, 0)
        self.touched = []

    def program_id(self, axis):
        return self.pid[axis]

    @staticmethod
    def arange(start, stop):
        return numpy.arange(start, stop, dtype=numpy.int64)

    @staticmethod
    def zeros(shape, dtype):
        return numpy.zeros(shape, dtype=dtype)

    def load(self, pointer, mask, other):
        offsets = numpy.broadcast_to(pointer.offset, numpy.shape(pointer.offset))
        lanes = numpy.broadcast_to(mask, offsets.shape)
        out = numpy.full(offsets.shape, other, dtype=numpy.float32)
        out[lanes] = pointer.base[offsets[lanes]]
        self.touched.append(offsets[lanes])
        return out

    def store(self, pointer, value, mask=True):
        offsets = numpy.asarray(pointer.offset)
        lanes = numpy.broadcast_to(mask, offsets.shape)
        values = numpy.broadcast_to(value, offsets.shape)
        pointer.base[offsets[lanes]] = values[lanes]

    @staticmethod
    def dot(a, b, acc=None, input_precision=None):
        assert input_precision == "ieee"
        assert a.dtype == b.dtype == numpy.float32
        base = numpy.float32(0.0) if acc is None else acc
        return numpy.matmul(a, b) + base

    @staticmethod
    def where(condition, x, y):
        return numpy.where(condition, x, numpy.float32(y)).astype(numpy.float32)

    @staticmethod
    def max(x, axis):
        return x.max(axis=axis)

    @staticmethod
    def sum(x, axis):
        return x.sum(axis=axis)

    maximum = staticmethod(numpy.maximum)
    exp = staticmethod(numpy.exp)


class Kernel:
    def __init__(self, fn, language):
        self.fn, self.language = fn, language

    def __getitem__(self, grid):
        def launch(*args, num_warps, num_stages, **constexprs):
            for pid_bh in range(grid[1]):
                for pid_m in range(grid[0]):
                    self.language.pid = (pid_m, pid_bh)
                    self.fn(*args, **constexprs)

        return launch


def install(monkeypatch):
    """Put the model in ``sys.modules`` for one test; return its language."""

    language = Language()
    triton = types.ModuleType("triton")
    triton.language = language
    triton.jit = lambda fn: Kernel(fn, language)
    triton.cdiv = lambda a, b: -(-a // b)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    return language


def compile_emulated(text):
    namespace = {"__name__": "emulated_flash_text"}
    exec(compile(text, "<flash text>", "exec"), namespace)
    return namespace


@pytest.fixture
def emulated(monkeypatch):
    language = install(monkeypatch)
    return compile_emulated(TRITON_SOURCE), language


def run_text(namespace, q, k, v, causal, config, buffers=None):
    heads, sequence, dim = q.shape
    rows = padded_rows(sequence, config)
    if buffers is None:
        buffers = (
            numpy.full(heads * rows * dim, SENTINEL).view(numpy.float32),
            numpy.full(heads * rows, SENTINEL).view(numpy.float32),
            numpy.full(heads * rows, SENTINEL).view(numpy.float32),
        )
    o, maxima, sums = buffers
    namespace[LAUNCHER_NAME](
        Pointer(q.reshape(-1)),
        Pointer(k.reshape(-1)),
        Pointer(v.reshape(-1)),
        Pointer(o),
        Pointer(maxima),
        Pointer(sums),
        heads,
        sequence,
        dim,
        float32_scale(dim),
        rows,
        causal=causal,
        **config.launcher_arguments(),
    )
    return (
        o.reshape(heads, rows, dim),
        maxima.reshape(heads, rows),
        sums.reshape(heads, rows),
    )


class EmulatedBackend:
    """A harness backend that runs one text under the model."""

    environment = {}

    def __init__(self, text=TRITON_SOURCE):
        self.namespace = compile_emulated(text)

    def run(self, q, k, v, *, causal, config, output, maxima, sums):
        run_text(self.namespace, q, k, v, causal, config, (output, maxima, sums))
        return output, maxima, sums


CASES = (
    AttentionCase(1, 1, 16, False),
    AttentionCase(2, 17, 16, True),
    AttentionCase(2, 70, 32, False),
    AttentionCase(1, 129, 64, True),
)


@pytest.mark.parametrize("config", FLASH_CONFIGS, ids=lambda config: config.name)
@pytest.mark.parametrize("family", FAMILIES)
def test_the_text_gives_the_bits_of_the_blockwise_reference(emulated, config, family):
    namespace, language = emulated
    for index, case in enumerate(CASES):
        q, k, v = attention_inputs(family, 74, index, case)
        o, maxima, sums = run_text(namespace, q, k, v, case.causal, config)
        expected = blockwise_attention(
            q, k, v, case.causal, config.block_m, config.block_n
        )
        n = case.sequence

        for got, want in zip((o[:, :n], maxima[:, :n], sums[:, :n]), expected):
            assert numpy.array_equal(got.view(numpy.uint32), want.view(numpy.uint32))
        # Rows past N keep their sentinels: every store kept its mask.
        for guard in (o[:, n:], maxima[:, n:], sums[:, n:]):
            assert (guard.view(numpy.uint32) == SENTINEL).all()
        # Every load stayed inside its head's [N, D] block.
        assert all(
            (offsets >= 0).all() and (offsets < q.size).all()
            for offsets in language.touched
        )
        language.touched.clear()


def test_the_grid_has_one_program_per_query_block_and_head(emulated, monkeypatch):
    namespace, language = emulated
    seen = []
    kernel = namespace[KERNEL_NAME]
    original = kernel.fn

    def spy(*args, **kwargs):
        seen.append(language.pid)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "fn", spy)
    case = AttentionCase(3, 70, 32, True)
    q, k, v = attention_inputs("normal", 74, 0, case)
    run_text(namespace, q, k, v, True, FLASH_CONFIGS[2])

    assert sorted(seen) == [(m, h) for m in range(math.ceil(70 / 64)) for h in range(3)]


def test_a_text_without_the_key_mask_reads_past_the_end_of_k(emulated, monkeypatch):
    namespace, language = emulated
    broken = TRITON_SOURCE.replace(
        "kt = tl.load(kt_ptrs, mask=col_mask[None, :], other=0.0)",
        "kt = tl.load(kt_ptrs, mask=offs_n[None, :] >= 0, other=0.0)",
    )
    assert broken != TRITON_SOURCE
    exec(compile(broken, "<broken flash text>", "exec"), namespace)
    case = AttentionCase(1, 17, 16, False)
    q, k, v = attention_inputs("normal", 74, 0, case)

    with pytest.raises(IndexError):
        run_text(namespace, q, k, v, False, FLASH_CONFIGS[3])
