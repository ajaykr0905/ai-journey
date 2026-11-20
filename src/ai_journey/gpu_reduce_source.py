"""Metal Shading Language source of the Day 78 float32 sum reductions R0 to R5.

GPU MODE Lecture 9 builds a sum reduction in steps, each one change on the
step before it. This module holds one kernel per step, each with its own
digest, plus one combine kernel that every step shares:

- R0: one element per thread, then a tree in threadgroup memory with
  interleaved addressing. At stride ``s`` a thread adds only when
  ``t % (2 s) == 0``, so the active lanes of a SIMD group are spread out.
- R1: sequential addressing. At stride ``s`` the threads ``t < s`` add, and
  ``s`` halves, so while ``s`` is 32 or more a whole SIMD group takes the
  branch or skips it.
- R2: R1 with the first add during the global load. Each thread adds two
  elements 256 apart, so one group covers 512 elements.
- R3: R2 with the last 32 values added in SIMD group 0 by
  ``simd_shuffle_down`` at deltas 16, 8, 4, 2 and 1, with no barrier. The
  shuffles add the same pairs as the R2 tree, in the same order.
- R4: R2's two loads, then ``simd_sum`` in each SIMD group, one barrier, and
  ``simd_sum`` of the 8 group sums in SIMD group 0.
- R5: R4 with a grid-stride loop over a fixed grid of 1024 groups, so the
  same grid covers any n.

SIMD-group functions are Metal's counterpart of CUDA warp shuffles, not the
same instructions. Each pass-1 group writes one partial sum. The second pass
is always ``reduce_combine``: one group of 1024 threads adds every partial
and writes the result, so no kernel needs an atomic and the launch fixes
the order of every addition.

These rules keep every kernel correct and its bound valid:

- Every barrier is outside any condition on the thread, and loop bounds
  depend only on the group size, so every thread of a group reaches every
  barrier.
- A load at or past ``n`` gives ``0.0f``, and an addition of 0 is exact.
- A SIMD-group function runs only where all 32 lanes of its SIMD group are
  active, and only kernels that need it ask for 32-lane SIMD groups.

The tree loops of R0 to R3 and of the combine take their bound from
``threads_per_threadgroup``, as the lecture's CUDA code takes it from
``blockDim.x``. The compiler thus keeps the modulo of R0, and the runtime
launches each kernel only with its declared group size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    require_integer,
)

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"
FLOAT32_BYTES = 4
GROUP_SIZE = 256
SIMD_WIDTH = 32
SIMD_GROUPS = GROUP_SIZE // SIMD_WIDTH
COMBINE_GROUP_SIZE = 1024
COMBINE_ACCUMULATORS = 4
FIXED_GROUPS = 1024
LOADS_PER_THREAD = 2
# The scalar n (or the partial count of the combine) is buffer 2 of every
# kernel, a 32-bit constant uint&.
SCALAR_BUFFERS = (2,)

R0 = "reduce_r0_interleaved"
R1 = "reduce_r1_sequential"
R2 = "reduce_r2_first_add"
R3 = "reduce_r3_simd_shuffle"
R4 = "reduce_r4_simd_sum"
R5 = "reduce_r5_grid_stride"
COMBINE = "reduce_combine"
STEP_KERNELS = (R0, R1, R2, R3, R4, R5)

# How each pass-1 kernel maps elements to threads.
ONE_PER_THREAD = "one_element_per_thread"
TWO_PER_THREAD = "two_elements_256_apart"
GRID_STRIDE = "grid_stride_loop"
COMBINE_STRIDE = "grid_stride_in_one_group"
MAPPINGS = (ONE_PER_THREAD, TWO_PER_THREAD, GRID_STRIDE, COMBINE_STRIDE)

# How the values of one group become one sum.
INTERLEAVED_TREE = "interleaved_tree"
SEQUENTIAL_TREE = "sequential_tree"
TREE_THEN_SHUFFLE = "sequential_tree_then_simd_shuffle_down"
SIMD_SUM = "simd_sum_then_simd_sum"
TREES = (INTERLEAVED_TREE, SEQUENTIAL_TREE, TREE_THEN_SHUFFLE, SIMD_SUM)

_NAME = re.compile(r"reduce_[a-z0-9_]{1,40}")


class GpuReduceError(KernelMetalError):
    """Raised when a reduction kernel, control, launch, check or report is invalid."""


def require_count(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer by the Metal lab's rule, as this lab's error.

    ``require_integer`` refuses ``true`` and ``2.0`` with ``TypeError``; a
    value out of range becomes ``GpuReduceError``.
    """

    try:
        return require_integer(value, name, minimum, maximum)
    except GpuReduceError:
        raise
    except KernelMetalError as error:
        raise GpuReduceError(str(error)) from error


def text_sha256(text: str) -> str:
    """Hash the exact ASCII text that ``compile_shader`` receives."""

    return sha256(text.encode("ascii")).hexdigest()


def _signature(name: str, parameters: tuple[str, ...]) -> list[str]:
    """Align each parameter under the first, as the earlier Metal labs do."""

    head = f"kernel void {name}("
    separator = ",\n" + " " * len(head)
    return (head + separator.join(parameters) + ") {").splitlines()


_PASS_ONE = (
    "device float* partials [[buffer(0)]]",
    "device const float* x [[buffer(1)]]",
    "constant uint& n [[buffer(2)]]",
    "uint group [[threadgroup_position_in_grid]]",
    "uint t [[thread_index_in_threadgroup]]",
)
_SIZE = "uint size [[threads_per_threadgroup]]"
_LANES = (
    "uint lane [[thread_index_in_simdgroup]]",
    "uint simd_group [[simdgroup_index_in_threadgroup]]",
)


def _kernel(comment: list[str], name: str, parameters, body: list[str]) -> str:
    lines = [*(f"// {line}" for line in comment), *_signature(name, parameters)]
    return "\n".join([*lines, *(f"    {line}" if line else "" for line in body), "}"])


def _load_one() -> list[str]:
    return [
        f"uint i = group * {GROUP_SIZE}u + t;",
        "shared[t] = i < n ? x[i] : 0.0f;",
        BARRIER,
    ]


def _first_add() -> str:
    """The R2 load: two elements 256 apart, added before the tree."""

    return (
        f"float v = (i < n ? x[i] : 0.0f) + "
        f"(i + {GROUP_SIZE}u < n ? x[i + {GROUP_SIZE}u] : 0.0f);"
    )


def _load_two() -> list[str]:
    return [
        f"uint i = group * {LOADS_PER_THREAD * GROUP_SIZE}u + t;",
        _first_add(),
        "shared[t] = v;",
        BARRIER,
    ]


def _sequential(stop: str) -> list[str]:
    return [
        f"for (uint s = size / 2u; s {stop}; s >>= 1u) {{",
        "    if (t < s) {",
        "        shared[t] += shared[t + s];",
        "    }",
        f"    {BARRIER}",
        "}",
    ]


def _write_partial() -> list[str]:
    return ["if (t == 0u) {", "    partials[group] = shared[0];", "}"]


def _simd_groups() -> list[str]:
    """Add each SIMD group with simd_sum, then the 8 group sums in group 0."""

    return [
        "v = simd_sum(v);",
        "if (lane == 0u) {",
        "    shared[simd_group] = v;",
        "}",
        BARRIER,
        "if (simd_group == 0u) {",
        f"    float w = lane < {SIMD_GROUPS}u ? shared[lane] : 0.0f;",
        "    w = simd_sum(w);",
        "    if (lane == 0u) {",
        "        partials[group] = w;",
        "    }",
        "}",
    ]


_R0 = _kernel(
    [
        "R0: one element per thread, then a tree with interleaved addressing.",
        "At stride s only threads with t % (2 s) == 0 add, so the active lanes",
        "of every SIMD group are spread out until s reaches 32.",
    ],
    R0,
    (*_PASS_ONE, _SIZE),
    [
        f"threadgroup float shared[{GROUP_SIZE}];",
        *_load_one(),
        "for (uint s = 1u; s < size; s *= 2u) {",
        "    if (t % (2u * s) == 0u) {",
        "        shared[t] += shared[t + s];",
        "    }",
        f"    {BARRIER}",
        "}",
        *_write_partial(),
    ],
)
_R1 = _kernel(
    [
        "R1: the R0 load, then a tree with sequential addressing: threads",
        "t < s add shared[t + s], and s halves. While s is 32 or more, a whole",
        "SIMD group takes the branch or skips it.",
    ],
    R1,
    (*_PASS_ONE, _SIZE),
    [
        f"threadgroup float shared[{GROUP_SIZE}];",
        *_load_one(),
        *_sequential("> 0u"),
        *_write_partial(),
    ],
)
_R2 = _kernel(
    [
        "R2: each thread adds two elements 256 apart during the load, so one",
        "group covers 512 elements; then the R1 tree.",
    ],
    R2,
    (*_PASS_ONE, _SIZE),
    [
        f"threadgroup float shared[{GROUP_SIZE}];",
        *_load_two(),
        *_sequential("> 0u"),
        *_write_partial(),
    ],
)
_R3 = _kernel(
    [
        "R3: the R2 tree stops at 32 values; SIMD group 0 adds them with",
        "simd_shuffle_down at deltas 16, 8, 4, 2 and 1, with no barrier. The",
        "shuffles add the same pairs as the R2 tree, in the same order.",
    ],
    R3,
    (*_PASS_ONE, _SIZE, *_LANES),
    [
        f"threadgroup float shared[{GROUP_SIZE}];",
        *_load_two(),
        *_sequential(f">= {SIMD_WIDTH}u"),
        "if (simd_group == 0u) {",
        "    float w = shared[lane];",
        *(f"    w += simd_shuffle_down(w, {delta}u);" for delta in (16, 8, 4, 2, 1)),
        "    if (lane == 0u) {",
        "        partials[group] = w;",
        "    }",
        "}",
    ],
)
_R4 = _kernel(
    [
        "R4: the R2 load, then simd_sum in each SIMD group, one barrier, and",
        f"simd_sum of the {SIMD_GROUPS} group sums in SIMD group 0.",
    ],
    R4,
    (*_PASS_ONE, *_LANES),
    [
        f"threadgroup float shared[{SIMD_GROUPS}];",
        f"uint i = group * {LOADS_PER_THREAD * GROUP_SIZE}u + t;",
        _first_add(),
        *_simd_groups(),
    ],
)
_R5 = _kernel(
    [
        f"R5: a fixed grid of {FIXED_GROUPS} groups. Each thread adds elements gid,",
        "gid + grid, gid + 2 grid, ... in one loop, then reduces as R4 does,",
        "so the same grid covers any n.",
    ],
    R5,
    (
        *_PASS_ONE[:4],
        "uint gid [[thread_position_in_grid]]",
        "uint grid [[threads_per_grid]]",
        *_LANES,
    ),
    [
        f"threadgroup float shared[{SIMD_GROUPS}];",
        "float v = 0.0f;",
        "for (uint e = gid; e < n; e += grid) {",
        "    v += x[e];",
        "}",
        *_simd_groups(),
    ],
)
_STRIDE = COMBINE_GROUP_SIZE
_COMBINE = _kernel(
    [
        f"Pass 2 of every step: one group of {COMBINE_GROUP_SIZE} threads. Thread t adds",
        f"partials t, t + {_STRIDE}, ... in {COMBINE_ACCUMULATORS} accumulators, "
        "then the group adds",
        "the thread sums with sequential addressing and writes out[0].",
    ],
    COMBINE,
    (
        "device float* out [[buffer(0)]]",
        "device const float* partials [[buffer(1)]]",
        "constant uint& count [[buffer(2)]]",
        "uint t [[thread_index_in_threadgroup]]",
        _SIZE,
    ),
    [
        f"threadgroup float shared[{COMBINE_GROUP_SIZE}];",
        "float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;",
        "uint e = t;",
        f"for (; e + {3 * _STRIDE}u < count; e += {4 * _STRIDE}u) {{",
        "    a0 += partials[e];",
        f"    a1 += partials[e + {_STRIDE}u];",
        f"    a2 += partials[e + {2 * _STRIDE}u];",
        f"    a3 += partials[e + {3 * _STRIDE}u];",
        "}",
        f"for (; e < count; e += {_STRIDE}u) {{",
        "    a0 += partials[e];",
        "}",
        "shared[t] = (a0 + a1) + (a2 + a3);",
        BARRIER,
        *_sequential("> 0u"),
        "if (t == 0u) {",
        "    out[0] = shared[0];",
        "}",
    ],
)


@dataclass(frozen=True)
class ReduceKernel:
    """One compiled kernel: its text, launch shape and what it declares."""

    name: str
    step: str | None
    mapping: str
    tree: str
    group_size: int
    threadgroup_floats: int
    simd_lanes: int | None
    fixed_groups: int | None
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuReduceError("a kernel name must be reduce_ and a short slug")
        if self.step is not None and self.step not in {
            f"R{index}" for index in range(len(STEP_KERNELS))
        }:
            raise GpuReduceError("a step is R0 to R5, or None for the combine pass")
        if self.mapping not in MAPPINGS or self.tree not in TREES:
            raise GpuReduceError(f"{self.name} has an unknown mapping or tree")
        require_count(self.group_size, "group_size", 1, MAX_THREADGROUP_THREADS)
        require_count(self.threadgroup_floats, "threadgroup_floats", 1, 1 << 13)
        if self.simd_lanes not in (None, SIMD_WIDTH):
            raise GpuReduceError(f"simd_lanes must be None or {SIMD_WIDTH}")
        if self.fixed_groups is not None:
            require_count(self.fixed_groups, "fixed_groups", 1, 1 << 16)
        if (self.fixed_groups is not None) != (self.mapping == GRID_STRIDE):
            raise GpuReduceError("exactly a grid-stride kernel has a fixed grid")
        if (
            not isinstance(self.source, str)
            or not self.source.isascii()
            or f"kernel void {self.name}(" not in self.source
            or "atomic" in self.source
        ):
            raise GpuReduceError(
                f"{self.name} needs ASCII source that defines it, with no atomic"
            )

    @property
    def threadgroup_bytes(self) -> int:
        """Static threadgroup memory that the kernel declares."""

        return self.threadgroup_floats * FLOAT32_BYTES

    @property
    def source_sha256(self) -> str:
        """The digest of this kernel's text alone."""

        return text_sha256(self.source)

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass the scalar as 32 bits, the width of ``constant uint&``."""

        return {index: "int32" for index in SCALAR_BUFFERS}

    def to_mapping(self) -> dict[str, object]:
        """Return what a report records about the kernel, without its text."""

        return {
            "name": self.name,
            "step": self.step,
            "mapping": self.mapping,
            "tree": self.tree,
            "group_size": self.group_size,
            "threadgroup_bytes": self.threadgroup_bytes,
            "simd_lanes": self.simd_lanes,
            "fixed_groups": self.fixed_groups,
            "source_sha256": self.source_sha256,
        }


KERNELS: dict[str, ReduceKernel] = {
    kernel.name: kernel
    for kernel in (
        ReduceKernel(
            R0, "R0", ONE_PER_THREAD, INTERLEAVED_TREE, GROUP_SIZE, 256, None, None, _R0
        ),
        ReduceKernel(
            R1, "R1", ONE_PER_THREAD, SEQUENTIAL_TREE, GROUP_SIZE, 256, None, None, _R1
        ),
        ReduceKernel(
            R2, "R2", TWO_PER_THREAD, SEQUENTIAL_TREE, GROUP_SIZE, 256, None, None, _R2
        ),
        ReduceKernel(
            R3,
            "R3",
            TWO_PER_THREAD,
            TREE_THEN_SHUFFLE,
            GROUP_SIZE,
            256,
            SIMD_WIDTH,
            None,
            _R3,
        ),
        ReduceKernel(
            R4,
            "R4",
            TWO_PER_THREAD,
            SIMD_SUM,
            GROUP_SIZE,
            SIMD_GROUPS,
            SIMD_WIDTH,
            None,
            _R4,
        ),
        ReduceKernel(
            R5,
            "R5",
            GRID_STRIDE,
            SIMD_SUM,
            GROUP_SIZE,
            SIMD_GROUPS,
            SIMD_WIDTH,
            FIXED_GROUPS,
            _R5,
        ),
        ReduceKernel(
            COMBINE,
            None,
            COMBINE_STRIDE,
            SEQUENTIAL_TREE,
            COMBINE_GROUP_SIZE,
            COMBINE_GROUP_SIZE,
            None,
            None,
            _COMBINE,
        ),
    )
}


def library_source() -> str:
    """Join the header and every kernel, in registry order, into one text."""

    return HEADER + "".join("\n" + kernel.source + "\n" for kernel in KERNELS.values())


def library_sha256() -> str:
    """The digest of the whole library that the runtime compiles."""

    return text_sha256(library_source())


def step_kernel(step: str) -> ReduceKernel:
    """Return the pass-1 kernel of one step, ``R0`` to ``R5``."""

    for kernel in KERNELS.values():
        if kernel.step == step:
            return kernel
    raise GpuReduceError(f"{step!r} is not a reduction step")
