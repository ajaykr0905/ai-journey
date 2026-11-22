"""Triton text of the softmax steps, checked on the CPU by the Triton interpreter.

GPU MODE Lecture 18 writes fused kernels in Triton. Triton has no macOS
wheel and no Metal backend, so this lab cannot run Triton on the Apple GPU.
This module holds Triton text of three of the steps, one program per row:

- the unfused U0: a row max kernel, an exp-sum kernel and a normalize
  kernel, with the max and the sum in device buffers;
- the fused F1 and F2 pattern: one block holds the whole row, then
  ``tl.max``, ``tl.exp``, ``tl.sum`` and one store;
- the online F3 pattern: each lane keeps a running max and a running sum
  over the blocks of the row and rescales the sum when its max grows, then
  one merge and one write pass.

With ``TRITON_INTERPRET=1``, Triton runs each kernel on the CPU with NumPy.
The checks use the lab's float64 reference, gapped and ragged layouts and
sentinel gaps. NumPy sets the order of each ``tl.sum``, so the per-order
bounds of the Metal steps do not apply, and the checks use the sanity bound
``2**-16 * y_ref + 2**-125``. These runs are correctness checks only: no
Triton kernel runs on a GPU, and no time is measured. ``triton`` is an
optional test dependency; this module skips when it or the interpreter
flag is absent.
"""

import os
from pathlib import Path

import numpy
import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "the Triton harness runs only with TRITON_INTERPRET=1, on the CPU",
        allow_module_level=True,
    )
triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
tl = triton.language

from ai_journey.metal_fused_config import load_config  # noqa: E402
from ai_journey.metal_fused_reference import (  # noqa: E402
    CORRECTNESS_STREAM,
    SENTINEL_BITS,
    row_sum_errors,
    sentinel_buffer,
    softmax_inputs,
    softmax_reference,
)

PINNED = Path(__file__).resolve().parents[1] / "config/day-79-metal-fused.json"
BLOCK = 256
SANITY = 2.0**-16
FLOOR = 2.0**-125


@triton.jit
def row_max_kernel(x_ptr, starts_ptr, lengths_ptr, max_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row)
    length = tl.load(lengths_ptr + row)
    columns = tl.arange(0, BLOCK)
    top = tl.full([BLOCK], -3.4028234663852886e38, tl.float32)
    for first in range(0, length, BLOCK):
        mask = first + columns < length
        x = tl.load(x_ptr + start + first + columns, mask=mask, other=-3.4e38)
        top = tl.maximum(top, x)
    tl.store(max_ptr + row, tl.max(top, axis=0))


@triton.jit
def exp_sum_kernel(
    x_ptr, starts_ptr, lengths_ptr, max_ptr, sum_ptr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row)
    length = tl.load(lengths_ptr + row)
    top = tl.load(max_ptr + row)
    columns = tl.arange(0, BLOCK)
    total = tl.zeros([BLOCK], tl.float32)
    for first in range(0, length, BLOCK):
        mask = first + columns < length
        x = tl.load(x_ptr + start + first + columns, mask=mask, other=-3.4e38)
        total += tl.where(mask, tl.exp(x - top), 0.0)
    tl.store(sum_ptr + row, tl.sum(total, axis=0))


@triton.jit
def normalize_kernel(
    x_ptr, starts_ptr, lengths_ptr, max_ptr, sum_ptr, y_ptr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row)
    length = tl.load(lengths_ptr + row)
    top = tl.load(max_ptr + row)
    total = tl.load(sum_ptr + row)
    columns = tl.arange(0, BLOCK)
    for first in range(0, length, BLOCK):
        mask = first + columns < length
        x = tl.load(x_ptr + start + first + columns, mask=mask, other=-3.4e38)
        tl.store(y_ptr + start + first + columns, tl.exp(x - top) / total, mask=mask)


@triton.jit
def fused_kernel(x_ptr, starts_ptr, lengths_ptr, y_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row)
    length = tl.load(lengths_ptr + row)
    columns = tl.arange(0, BLOCK)
    mask = columns < length
    x = tl.load(x_ptr + start + columns, mask=mask, other=-3.4e38)
    top = tl.max(x, axis=0)
    terms = tl.where(mask, tl.exp(x - top), 0.0)
    tl.store(y_ptr + start + columns, terms / tl.sum(terms, axis=0), mask=mask)


@triton.jit
def online_kernel(x_ptr, starts_ptr, lengths_ptr, y_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(starts_ptr + row)
    length = tl.load(lengths_ptr + row)
    columns = tl.arange(0, BLOCK)
    running = tl.full([BLOCK], -3.4028234663852886e38, tl.float32)
    total = tl.zeros([BLOCK], tl.float32)
    for first in range(0, length, BLOCK):
        mask = first + columns < length
        x = tl.load(x_ptr + start + first + columns, mask=mask, other=-3.4e38)
        grown = tl.maximum(running, x)
        total = total * tl.exp(running - grown) + tl.where(mask, tl.exp(x - grown), 0.0)
        running = grown
    top = tl.max(running, axis=0)
    total = tl.sum(total * tl.exp(running - top), axis=0)
    for first in range(0, length, BLOCK):
        mask = first + columns < length
        x = tl.load(x_ptr + start + first + columns, mask=mask, other=-3.4e38)
        tl.store(y_ptr + start + first + columns, tl.exp(x - top) / total, mask=mask)


def _tensors(values, layout):
    return (
        torch.from_numpy(values.copy()),
        torch.from_numpy(layout.starts()),
        torch.from_numpy(layout.lengths_array()),
        torch.from_numpy(sentinel_buffer(layout).copy()),
    )


def run_unfused(values, layout):
    x, starts, lengths, y = _tensors(values, layout)
    top = torch.empty(layout.rows, dtype=torch.float32)
    total = torch.empty(layout.rows, dtype=torch.float32)
    grid = (layout.rows,)
    row_max_kernel[grid](x, starts, lengths, top, BLOCK=BLOCK)
    exp_sum_kernel[grid](x, starts, lengths, top, total, BLOCK=BLOCK)
    normalize_kernel[grid](x, starts, lengths, top, total, y, BLOCK=BLOCK)
    return y.numpy()


def run_fused(values, layout):
    x, starts, lengths, y = _tensors(values, layout)
    width = triton.next_power_of_2(layout.longest)
    fused_kernel[(layout.rows,)](x, starts, lengths, y, BLOCK=width)
    return y.numpy()


def run_online(values, layout):
    x, starts, lengths, y = _tensors(values, layout)
    online_kernel[(layout.rows,)](x, starts, lengths, y, BLOCK=BLOCK)
    return y.numpy()


RUNS = {"unfused": run_unfused, "fused": run_fused, "online": run_online}
CASES = load_config(PINNED).correctness.cases


@pytest.mark.parametrize("name", RUNS)
@pytest.mark.parametrize("index", range(len(CASES)))
def test_each_triton_step_matches_the_reference_in_the_interpreter(name, index):
    case = CASES[index]
    layout = case.layout()
    values = softmax_inputs(79, CORRECTNESS_STREAM, index, layout, case.inputs)
    reference = softmax_reference(values, layout)

    output = RUNS[name](values, layout)

    gaps = numpy.ones(layout.buffer_elements, dtype=bool)
    gaps[layout.valid()] = False
    assert numpy.all(output[gaps].view(numpy.uint32) == SENTINEL_BITS)
    packed = output[layout.valid()]
    assert numpy.all(numpy.isfinite(packed))
    error = numpy.abs(packed.astype(numpy.float64) - reference.output)
    assert numpy.all(error <= SANITY * reference.output + FLOOR)
    assert numpy.all(row_sum_errors(packed, layout) <= 2.0**-12)
