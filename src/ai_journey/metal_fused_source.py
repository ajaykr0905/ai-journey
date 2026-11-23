"""Metal Shading Language source of the Day 79 float32 row softmax kernels.

Curriculum Day 79 (Track B) follows GPU MODE Lecture 18, "Fused Kernels".
This module builds one softmax over the rows of a batch in four steps. Each
step makes one change to the step before it:

- U0, unfused: three kernels. ``softmax_u0_row_max`` writes the max of each
  row to a device buffer. ``softmax_u0_exp_sum`` reads the row and that max
  and writes the sum of ``exp(x - max)`` to a second buffer.
  ``softmax_u0_normalize`` reads the row, the max and the sum and writes
  ``exp(x - max) / sum``. Each of the three kernels reads the row from
  device memory.
- F1, fused: one kernel. Each thread loads its slots of the row once into
  registers. A tree in threadgroup memory gives the max, a second tree gives
  the sum, and then each thread writes its outputs.
- F2: F1 with SIMD-group reductions. ``simd_max`` and ``simd_sum`` reduce
  each SIMD group, and one threadgroup step joins the SIMD groups, once for
  the max and once for the sum.
- F3, online softmax: one pass over the slots keeps a running max and a
  running sum, and rescales the sum when the max grows (M. Milakov and N.
  Gimelshein, "Online normalizer calculation for softmax", 2018). One
  threadgroup step merges the (max, sum) pairs, and one pass writes.

Every kernel runs one threadgroup per row. The group size G is a power of
two from 32 to 1024, and thread ``lane`` owns the columns ``lane + i * G``
for ``i < SLOTS``, so a launch needs ``G * SLOTS`` at least the longest row.
The row of group ``r`` starts at ``starts[r]`` and holds ``lengths[r]``
elements, so dense, packed ragged and gapped layouts use the same code.

These rules keep each kernel correct and its error bound valid:

- Every barrier is outside any condition on the thread. A loop around a
  barrier takes its bound from G only, so every thread reaches every
  barrier. ``require_uniform_barriers`` checks this rule on the text.
- A column at or past the row's length is never read and never written.
  An idle slot holds ``LOWEST``, the most negative finite float32, never an
  infinity, so no subtraction makes a NaN.
- ``precise::exp`` and ``precise::divide`` fix the math functions whatever
  the compile options are. The max subtraction keeps every ``exp`` argument
  at or below 0, so no ``exp`` overflows.

Each kernel and each step has a SHA-256 digest of its exact text, and the
library joins every kernel under one header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    require_integer,
)
from .matmul_tiled_source import BARRIER, HEADER
from .matmul_tiled_variants import text_sha256

FLOAT32_BYTES = 4
UINT32_BYTES = 4
SLOTS = 16
SIMD_WIDTH = 32
MIN_GROUP = SIMD_WIDTH
MAX_GROUP = MAX_THREADGROUP_THREADS
MAX_ROW_LENGTH = SLOTS * MAX_GROUP
# The most negative finite float32, as MSL text and as its value.
LOWEST_TEXT = "-3.40282347e+38f"
LOWEST = -3.4028234663852886e38
EXP_PROBE_GROUP = 256

U0 = "softmax_u0_unfused"
F1 = "softmax_f1_threadgroup"
F2 = "softmax_f2_simd"
F3 = "softmax_f3_online"
STEP_VARIANTS = (U0, F1, F2, F3)
U0_ROW_MAX = "softmax_u0_row_max"
U0_EXP_SUM = "softmax_u0_exp_sum"
U0_NORMALIZE = "softmax_u0_normalize"
EXP_PROBE = "softmax_exp_probe"

THREADGROUP_TREE = "threadgroup_tree"
SIMD_THEN_ONE_STEP = "simd_then_one_threadgroup_step"
ONLINE_SIMD_THEN_ONE_STEP = "online_simd_then_one_threadgroup_step"
REDUCTIONS = (THREADGROUP_TREE, SIMD_THEN_ONE_STEP, ONLINE_SIMD_THEN_ONE_STEP)
_TREE_LOOP = "for (uint stride = size / 2u; stride > 0u; stride >>= 1u) {"
_NAME = re.compile(r"softmax_[a-z0-9_]{1,40}")


class MetalFusedError(KernelMetalError):
    """Raised when a fused softmax kernel, control, launch, check or report is invalid."""


def require_count(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer as the Metal labs do, and raise this lab's error.

    ``require_integer`` refuses ``true`` and ``2.0`` with ``TypeError``; a
    value out of range becomes ``MetalFusedError``.
    """

    try:
        return require_integer(value, name, minimum, maximum)
    except MetalFusedError:
        raise
    except KernelMetalError as error:
        raise MetalFusedError(str(error)) from error


def group_size(longest: int) -> int:
    """Return the launch rule's G: the fewest threads, a power of two from 32,
    whose ``SLOTS`` slots each hold the longest row of the batch.
    """

    require_count(longest, "longest row", 1, MAX_ROW_LENGTH)
    size = MIN_GROUP
    while size * SLOTS < longest:
        size *= 2
    return size


def slots_used(length: int, size: int) -> int:
    """Return how many slots thread 0 fills for a row: ``ceil(length / G)``."""

    require_count(length, "row length", 1, MAX_ROW_LENGTH)
    require_count(size, "group size", MIN_GROUP, MAX_GROUP)
    used = -(-length // size)
    if used > SLOTS:
        raise MetalFusedError(f"a row of {length} needs more than {SLOTS} slots")
    return used


@dataclass(frozen=True)
class FusedKernel:
    """One kernel of the library: its buffers, threadgroup memory and text."""

    name: str
    buffers: tuple[str, ...]
    threadgroup_bytes: int
    simd_functions: bool
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalFusedError("a kernel name must be softmax_ and a short slug")
        if (
            type(self.buffers) is not tuple
            or not self.buffers
            or not all(isinstance(name, str) for name in self.buffers)
        ):
            raise MetalFusedError(f"{self.name} must name its buffers")
        require_count(self.threadgroup_bytes, "threadgroup_bytes", 0, 32 * 1024)
        if type(self.simd_functions) is not bool:
            raise TypeError("simd_functions must be a bool")
        if f"kernel void {self.name}(" not in self.source:
            raise MetalFusedError(f"{self.name} text does not define the kernel")

    @property
    def source_sha256(self) -> str:
        """Hash the exact text of this kernel."""

        return text_sha256(self.source)

    def to_mapping(self) -> dict[str, object]:
        """Return the kernel as a report records it."""

        return {
            "name": self.name,
            "buffers": list(self.buffers),
            "threadgroup_bytes": self.threadgroup_bytes,
            "simd_functions": self.simd_functions,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class SoftmaxVariant:
    """One step of the plan: its kernels, reduction and passes over the row."""

    step: str
    name: str
    kernels: tuple[str, ...]
    reduction: str
    row_reads: int
    summary: str

    def __post_init__(self) -> None:
        if self.name not in STEP_VARIANTS:
            raise MetalFusedError(f"{self.name!r} is not a registered step")
        if type(self.kernels) is not tuple or not self.kernels:
            raise MetalFusedError(f"{self.name} needs at least one kernel")
        if self.reduction not in REDUCTIONS:
            raise MetalFusedError(f"reduction must be one of {', '.join(REDUCTIONS)}")
        require_count(self.row_reads, "row_reads", 1, len(self.kernels) * SLOTS)
        if not isinstance(self.summary, str) or not self.summary.isascii():
            raise MetalFusedError(f"{self.name} summary must be ASCII text")

    @property
    def launches_per_call(self) -> int:
        """Kernel launches in one softmax call: 3 for U0, 1 for a fused step."""

        return len(self.kernels)

    @property
    def source(self) -> str:
        """The text of every kernel of the step, in launch order."""

        return "\n".join(KERNELS[name].source for name in self.kernels)

    @property
    def source_sha256(self) -> str:
        """Hash the exact text of the step's kernels."""

        return text_sha256(self.source)

    @property
    def simd_functions(self) -> bool:
        """Whether any kernel of the step calls a SIMD-group function."""

        return any(KERNELS[name].simd_functions for name in self.kernels)

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report records it."""

        return {
            "step": self.step,
            "name": self.name,
            "kernels": list(self.kernels),
            "launches_per_call": self.launches_per_call,
            "reduction": self.reduction,
            "row_reads": self.row_reads,
            "summary": self.summary,
            "source_sha256": self.source_sha256,
        }


def _signature(name: str, parameters: tuple[str, ...]) -> list[str]:
    """Align each parameter under the first, as the earlier Metal labs do."""

    head = f"kernel void {name}("
    separator = ",\n" + " " * len(head)
    return (head + separator.join(parameters) + ") {").splitlines()


_KINDS = {
    "y": "device float*",
    "row_max_out": "device float*",
    "row_sum_out": "device float*",
    "x": "device const float*",
    "row_max": "device const float*",
    "row_sum": "device const float*",
    "starts": "device const uint*",
    "lengths": "device const uint*",
}
_ROW_THREADS = (
    "uint row [[threadgroup_position_in_grid]]",
    "uint lane [[thread_position_in_threadgroup]]",
    "uint size [[threads_per_threadgroup]]",
)
_SIMD_THREADS = (
    "uint simd_lane [[thread_index_in_simdgroup]]",
    "uint simd_group [[simdgroup_index_in_threadgroup]]",
)


def _parameters(buffers: tuple[str, ...], simd: bool) -> tuple[str, ...]:
    bound = tuple(
        f"{_KINDS[name]} {name.removesuffix('_out')} [[buffer({index})]]"
        for index, name in enumerate(buffers)
    )
    return (*bound, *_ROW_THREADS, *(_SIMD_THREADS if simd else ()))


def _kernel(comment: list[str], name: str, buffers, simd: bool, body) -> str:
    lines = [
        *(f"// {line}" for line in comment),
        *_signature(name, _parameters(buffers, simd)),
    ]
    return "\n".join([*lines, *(f"    {line}" for line in body), "}"]) + "\n"


_ROW_BOUNDS = ("uint start = starts[row];", "uint length = lengths[row];")


def _tree(array: str, combine: str) -> list[str]:
    """A tree over the G values of ``array``; the loop bound depends on G only."""

    if combine == "max":
        step = f"{array}[lane] = max({array}[lane], {array}[lane + stride]);"
    else:
        step = f"{array}[lane] += {array}[lane + stride];"
    return [
        _TREE_LOOP,
        "    if (lane < stride) {",
        f"        {step}",
        "    }",
        f"    {BARRIER}",
        "}",
    ]


def _slots(lines: list[str]) -> list[str]:
    """A loop over the thread's SLOTS columns ``c = lane + i * G``."""

    return [
        f"for (uint i = 0u; i < {SLOTS}u; ++i) {{",
        "    uint c = lane + i * size;",
        *(f"    {line}" for line in lines),
        "}",
    ]


def _load() -> list[str]:
    """Load the row once into registers and take the thread's max."""

    return [
        f"float v[{SLOTS}];",
        f"float m = {LOWEST_TEXT};",
        *_slots(
            [
                f"v[i] = c < length ? x[start + c] : {LOWEST_TEXT};",
                "m = max(m, v[i]);",
            ]
        ),
    ]


def _exponentials() -> list[str]:
    """Replace each slot by exp(v - max) and add the slots in order."""

    return [
        "float s = 0.0f;",
        *_slots(
            [
                "v[i] = c < length ? precise::exp(v[i] - row_max) : 0.0f;",
                "s += v[i];",
            ]
        ),
    ]


def _write(value: str) -> list[str]:
    return _slots(
        [
            "if (c < length) {",
            f"    y[start + c] = precise::divide({value}, total);",
            "}",
        ]
    )


_GROUP_FLOATS = MAX_GROUP * FLOAT32_BYTES
_PARTIAL_FLOATS = (MAX_GROUP // SIMD_WIDTH) * FLOAT32_BYTES

_U0_ROW_MAX = _kernel(
    [
        "U0 pass 1 of 3: the max of each row into row_max, one group per row.",
        "Each thread takes the max of its columns, then a tree in threadgroup",
        "memory takes the max of the G thread values.",
    ],
    U0_ROW_MAX,
    ("row_max_out", "x", "starts", "lengths"),
    False,
    [
        f"threadgroup float shared[{MAX_GROUP}];",
        *_ROW_BOUNDS,
        f"float m = {LOWEST_TEXT};",
        "for (uint c = lane; c < length; c += size) {",
        "    m = max(m, x[start + c]);",
        "}",
        "shared[lane] = m;",
        BARRIER,
        *_tree("shared", "max"),
        "if (lane == 0u) {",
        "    row_max[row] = shared[0];",
        "}",
    ],
)
_U0_EXP_SUM = _kernel(
    [
        "U0 pass 2 of 3: read the row and its max again and write the sum of",
        "exp(x - max) into row_sum. Each thread adds its columns in order, then",
        "a tree in threadgroup memory adds the G thread sums.",
    ],
    U0_EXP_SUM,
    ("row_sum_out", "x", "row_max", "starts", "lengths"),
    False,
    [
        f"threadgroup float shared[{MAX_GROUP}];",
        *_ROW_BOUNDS,
        "float m = row_max[row];",
        "float s = 0.0f;",
        "for (uint c = lane; c < length; c += size) {",
        "    s += precise::exp(x[start + c] - m);",
        "}",
        "shared[lane] = s;",
        BARRIER,
        *_tree("shared", "sum"),
        "if (lane == 0u) {",
        "    row_sum[row] = shared[0];",
        "}",
    ],
)
_U0_NORMALIZE = _kernel(
    [
        "U0 pass 3 of 3: read the row, its max and its sum a third time and",
        "write exp(x - max) / sum.",
    ],
    U0_NORMALIZE,
    ("y", "x", "row_max", "row_sum", "starts", "lengths"),
    False,
    [
        *_ROW_BOUNDS,
        "float m = row_max[row];",
        "float total = row_sum[row];",
        "for (uint c = lane; c < length; c += size) {",
        "    y[start + c] = precise::divide(precise::exp(x[start + c] - m), total);",
        "}",
    ],
)
_F1 = _kernel(
    [
        "F1: one kernel. Each thread loads its slots once into registers; a",
        "threadgroup tree gives the max, a second tree gives the sum, and each",
        "thread writes its slots. The additions are those of U0, in its order.",
    ],
    F1,
    ("y", "x", "starts", "lengths"),
    False,
    [
        f"threadgroup float max_shared[{MAX_GROUP}];",
        f"threadgroup float sum_shared[{MAX_GROUP}];",
        *_ROW_BOUNDS,
        *_load(),
        "max_shared[lane] = m;",
        BARRIER,
        *_tree("max_shared", "max"),
        "float row_max = max_shared[0];",
        *_exponentials(),
        "sum_shared[lane] = s;",
        BARRIER,
        *_tree("sum_shared", "sum"),
        "float total = sum_shared[0];",
        *_write("v[i]"),
    ],
)
_F2 = _kernel(
    [
        "F2: F1 with simd_max and simd_sum in each SIMD group and one",
        "threadgroup step for each value: lane 0 of each SIMD group writes its",
        "partial, and after one barrier every SIMD group reduces the partials.",
    ],
    F2,
    ("y", "x", "starts", "lengths"),
    True,
    [
        f"threadgroup float max_partial[{MAX_GROUP // SIMD_WIDTH}];",
        f"threadgroup float sum_partial[{MAX_GROUP // SIMD_WIDTH}];",
        *_ROW_BOUNDS,
        f"uint simd_groups = size / {SIMD_WIDTH}u;",
        *_load(),
        "m = simd_max(m);",
        "if (simd_lane == 0u) {",
        "    max_partial[simd_group] = m;",
        "}",
        BARRIER,
        "float row_max = simd_max(",
        f"    simd_lane < simd_groups ? max_partial[simd_lane] : {LOWEST_TEXT});",
        *_exponentials(),
        "s = simd_sum(s);",
        "if (simd_lane == 0u) {",
        "    sum_partial[simd_group] = s;",
        "}",
        BARRIER,
        "float total = simd_sum(",
        "    simd_lane < simd_groups ? sum_partial[simd_lane] : 0.0f);",
        *_write("v[i]"),
    ],
)
_F3 = _kernel(
    [
        "F3: online softmax. One pass keeps a running max m and a running sum d",
        "of exp(v - m); when v > m the old sum is rescaled by exp(m - v). Each",
        "SIMD group merges its (m, d) pairs, one threadgroup step merges the",
        "SIMD groups, and one pass writes exp(v - max) / sum.",
    ],
    F3,
    ("y", "x", "starts", "lengths"),
    True,
    [
        f"threadgroup float max_partial[{MAX_GROUP // SIMD_WIDTH}];",
        f"threadgroup float sum_partial[{MAX_GROUP // SIMD_WIDTH}];",
        *_ROW_BOUNDS,
        f"uint simd_groups = size / {SIMD_WIDTH}u;",
        f"float v[{SLOTS}];",
        f"float m = {LOWEST_TEXT};",
        "float d = 0.0f;",
        *_slots(
            [
                f"v[i] = c < length ? x[start + c] : {LOWEST_TEXT};",
                "if (c < length) {",
                "    float scale = precise::exp(min(m, v[i]) - max(m, v[i]));",
                "    d = v[i] > m ? fma(d, scale, 1.0f) : d + scale;",
                "    m = max(m, v[i]);",
                "}",
            ]
        ),
        "float simd_m = simd_max(m);",
        "float simd_d = simd_sum(d * precise::exp(m - simd_m));",
        "if (simd_lane == 0u) {",
        "    max_partial[simd_group] = simd_m;",
        "    sum_partial[simd_group] = simd_d;",
        "}",
        BARRIER,
        f"float group_m = simd_lane < simd_groups ? max_partial[simd_lane] : "
        f"{LOWEST_TEXT};",
        "float group_d = simd_lane < simd_groups ? sum_partial[simd_lane] : 0.0f;",
        "float row_max = simd_max(group_m);",
        "float total = simd_sum(group_d * precise::exp(group_m - row_max));",
        *_write("precise::exp(v[i] - row_max)"),
    ],
)
_EXP_PROBE = (
    "// The exp probe: y = precise::exp(x), one thread per argument, so a run\n"
    "// can measure the exp error that the bound assumes.\n"
    + "\n".join(
        _signature(
            EXP_PROBE,
            (
                "device float* y [[buffer(0)]]",
                "device const float* x [[buffer(1)]]",
                "uint i [[thread_position_in_grid]]",
            ),
        )
    )
    + "\n    y[i] = precise::exp(x[i]);\n}\n"
)

KERNELS = {
    kernel.name: kernel
    for kernel in (
        FusedKernel(
            U0_ROW_MAX,
            ("row_max", "x", "starts", "lengths"),
            _GROUP_FLOATS,
            False,
            _U0_ROW_MAX,
        ),
        FusedKernel(
            U0_EXP_SUM,
            ("row_sum", "x", "row_max", "starts", "lengths"),
            _GROUP_FLOATS,
            False,
            _U0_EXP_SUM,
        ),
        FusedKernel(
            U0_NORMALIZE,
            ("y", "x", "row_max", "row_sum", "starts", "lengths"),
            0,
            False,
            _U0_NORMALIZE,
        ),
        FusedKernel(F1, ("y", "x", "starts", "lengths"), 2 * _GROUP_FLOATS, False, _F1),
        FusedKernel(
            F2, ("y", "x", "starts", "lengths"), 2 * _PARTIAL_FLOATS, True, _F2
        ),
        FusedKernel(
            F3, ("y", "x", "starts", "lengths"), 2 * _PARTIAL_FLOATS, True, _F3
        ),
        FusedKernel(EXP_PROBE, ("y", "x"), 0, False, _EXP_PROBE),
    )
}
VARIANTS = {
    variant.name: variant
    for variant in (
        SoftmaxVariant(
            "U0",
            U0,
            (U0_ROW_MAX, U0_EXP_SUM, U0_NORMALIZE),
            THREADGROUP_TREE,
            3,
            "three kernels; the max and the sum go through device memory",
        ),
        SoftmaxVariant(
            "F1",
            F1,
            (F1,),
            THREADGROUP_TREE,
            1,
            "one kernel; the row stays in registers; two threadgroup trees",
        ),
        SoftmaxVariant(
            "F2",
            F2,
            (F2,),
            SIMD_THEN_ONE_STEP,
            1,
            "one kernel; simd_max and simd_sum, one threadgroup step for each",
        ),
        SoftmaxVariant(
            "F3",
            F3,
            (F3,),
            ONLINE_SIMD_THEN_ONE_STEP,
            1,
            "one kernel; one online pass for max and sum, one merge step",
        ),
    )
}


def require_uniform_barriers(text: str) -> int:
    """Check that no barrier of one kernel text sits inside a thread condition.

    The generated text puts each opening brace at the end of a line and each
    closing brace at the start of one. Every block around a barrier must be
    the kernel body or the tree loop, whose bound depends only on G. Return
    the count of barriers in the text.
    """

    blocks: list[str] = []
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("}"):
            if not blocks:
                raise MetalFusedError("a closing brace has no open block")
            blocks.pop()
        if line == BARRIER:
            count += 1
            inner = blocks[1:]
            if not blocks or not all(opener == _TREE_LOOP for opener in inner):
                raise MetalFusedError(
                    "a barrier sits inside a block that depends on the thread"
                )
        if line.endswith("{"):
            blocks.append(line)
    if blocks:
        raise MetalFusedError("a block of the kernel text is not closed")
    return count


def fused_library() -> str:
    """Join the header and every kernel into the one text that compiles."""

    return HEADER + "".join("\n" + kernel.source for kernel in KERNELS.values())


def library_sha256() -> str:
    """Hash the exact library text that ``compile_shader`` receives."""

    return text_sha256(fused_library())


for _kernel_entry in KERNELS.values():
    require_uniform_barriers(_kernel_entry.source)
