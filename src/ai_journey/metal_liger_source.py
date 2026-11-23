"""Metal Shading Language source of the Day 80 fused LLM-training kernels.

Curriculum Day 80 (Track B) follows GPU MODE Lecture 28, "Liger Kernel:
Efficient Triton Kernels for LLM Training". Liger Kernel writes Triton
kernels that fuse the small steps of common training layers. Triton has no
macOS build, so this lab writes three such forward kernels in Metal and runs
them on the Apple GPU through ``torch.mps.compile_shader``:

- ``liger_rmsnorm``: ``y = x / sqrt(mean(x^2) + eps) * w`` for each row.
  One threadgroup takes one row. Each thread keeps its slots of the row in
  registers, so the kernel reads ``x`` once. ``simd_sum`` and one
  threadgroup step give the sum of squares.
- ``liger_swiglu``: ``y = silu(a) * b``, element by element, in one pass.
- ``liger_cross_entropy``: the loss of each row, ``logsumexp(x) - x[t]``.
  One pass over the vocabulary keeps a running max and a running sum and
  rescales the sum when the max grows, as the Day 79 online softmax does.
  The kernel writes one float for each row and never writes the softmax.

The row of group ``r`` starts at ``starts[r]`` and holds ``lengths[r]``
elements, so dense, ragged and gapped layouts use the same code. These rules
keep each kernel correct and its declared error bound valid:

- Every barrier is outside any condition on the thread, so every thread
  reaches every barrier. The Day 79 ``require_uniform_barriers`` checks
  this rule on each kernel text.
- A column at or past the row's length is never read and never written.
- ``precise::`` math fixes the functions whatever the compile options are.
  Three probe kernels let a run measure the error of ``precise::exp``,
  ``precise::rsqrt`` and ``precise::log`` that the bounds assume.
- The cross-entropy kernel subtracts the target logit from the row max
  before it adds the log of the sum, so the rounding error does not grow
  with the magnitude of the logits.

Each kernel has a SHA-256 digest of its exact text, and the library joins
every kernel under one header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .matmul_tiled_source import BARRIER, HEADER
from .matmul_tiled_variants import text_sha256
from .metal_fused_source import (
    LOWEST_TEXT,
    MAX_GROUP,
    MIN_GROUP,
    SIMD_WIDTH,
    SLOTS,
    MetalFusedError,
    require_count,
    require_uniform_barriers,
)

FLOAT32_BYTES = 4
UINT32_BYTES = 4
# The register kernel holds SLOTS values for each of at most 1024 threads.
MAX_REGISTER_ROW = SLOTS * MAX_GROUP
# A streaming kernel loops over its row, so only the layout bounds a row.
MAX_STREAM_ROW = 1 << 17
EPS = 1.0e-6
EPS_TEXT = "1.0e-6f"
PROBE_GROUP = 256

RMSNORM = "rmsnorm"
SWIGLU = "swiglu"
CROSS_ENTROPY = "cross_entropy"
FAMILIES = (RMSNORM, SWIGLU, CROSS_ENTROPY)
RMSNORM_KERNEL = "liger_rmsnorm"
SWIGLU_KERNEL = "liger_swiglu"
CROSS_ENTROPY_KERNEL = "liger_cross_entropy"
FAMILY_KERNELS = {
    RMSNORM: RMSNORM_KERNEL,
    SWIGLU: SWIGLU_KERNEL,
    CROSS_ENTROPY: CROSS_ENTROPY_KERNEL,
}
PROBE_EXP = "liger_probe_exp"
PROBE_RSQRT = "liger_probe_rsqrt"
PROBE_LOG = "liger_probe_log"
PROBES = {"exp": PROBE_EXP, "rsqrt": PROBE_RSQRT, "log": PROBE_LOG}
_NAME = re.compile(r"liger_[a-z0-9_]{1,40}")


class MetalLigerError(MetalFusedError):
    """Raised when a Day 80 kernel, control, launch, check or report is invalid."""


def require_size(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer as the Metal labs do, and raise this lab's error."""

    try:
        return require_count(value, name, minimum, maximum)
    except MetalLigerError:
        raise
    except MetalFusedError as error:
        raise MetalLigerError(str(error)) from error


def launch_group(family: str, longest: int) -> int:
    """Return the group size G of one launch: one threadgroup for each row.

    G is the fewest threads, a power of two from 32, whose ``SLOTS`` slots
    each hold the longest row, but at most 1024. The register kernel needs
    every slot, so RMSNorm refuses a row longer than ``16 * 1024``. The
    streaming kernels loop over the row, so a longer row only gives each
    thread more columns.
    """

    if family not in FAMILIES:
        raise MetalLigerError(f"{family!r} is not a kernel family")
    limit = MAX_REGISTER_ROW if family == RMSNORM else MAX_STREAM_ROW
    require_size(longest, "longest row", 1, limit)
    size = MIN_GROUP
    while size * SLOTS < longest and size < MAX_GROUP:
        size *= 2
    return size


def columns_per_thread(length: int, size: int) -> int:
    """Return how many columns thread 0 takes from a row: ``ceil(length / G)``."""

    require_size(length, "row length", 1, MAX_STREAM_ROW)
    require_size(size, "group size", MIN_GROUP, MAX_GROUP)
    return -(-length // size)


@dataclass(frozen=True)
class LigerKernel:
    """One kernel of the library: its family, buffers, threadgroup bytes and text."""

    name: str
    family: str | None
    buffers: tuple[str, ...]
    threadgroup_bytes: int
    barriers: int
    summary: str
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalLigerError("a kernel name must be liger_ and a short slug")
        if self.family is not None and FAMILY_KERNELS.get(self.family) != self.name:
            raise MetalLigerError(f"{self.name} must name its own family")
        if (
            type(self.buffers) is not tuple
            or not self.buffers
            or not all(isinstance(name, str) for name in self.buffers)
        ):
            raise MetalLigerError(f"{self.name} must name its buffers")
        require_size(self.threadgroup_bytes, "threadgroup_bytes", 0, 32 * 1024)
        if not isinstance(self.summary, str) or not self.summary.isascii():
            raise MetalLigerError(f"{self.name} summary must be ASCII text")
        if f"kernel void {self.name}(" not in self.source:
            raise MetalLigerError(f"{self.name} text does not define the kernel")
        try:
            counted = require_uniform_barriers(self.source)
        except MetalFusedError as error:
            raise MetalLigerError(f"{self.name}: {error}") from error
        if counted != self.barriers:
            raise MetalLigerError(f"{self.name} declares the wrong barrier count")

    @property
    def source_sha256(self) -> str:
        """Hash the exact text of this kernel."""

        return text_sha256(self.source)

    def to_mapping(self) -> dict[str, object]:
        """Return the kernel as a report records it."""

        return {
            "name": self.name,
            "family": self.family,
            "buffers": list(self.buffers),
            "threadgroup_bytes": self.threadgroup_bytes,
            "barriers": self.barriers,
            "summary": self.summary,
            "source_sha256": self.source_sha256,
        }


_KINDS = {
    "y": "device float*",
    "loss": "device float*",
    "x": "device const float*",
    "w": "device const float*",
    "a": "device const float*",
    "b": "device const float*",
    "targets": "device const uint*",
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
_ROW_BOUNDS = (
    "uint start = starts[row];",
    "uint length = lengths[row];",
    f"uint simd_groups = size / {SIMD_WIDTH}u;",
)
_PARTIAL = MAX_GROUP // SIMD_WIDTH


def _kernel(comment, name: str, parameters, body) -> str:
    head = f"kernel void {name}("
    separator = ",\n" + " " * len(head)
    lines = [
        *(f"// {line}" for line in comment),
        *(head + separator.join(parameters) + ") {").splitlines(),
        *(f"    {line}" for line in body),
        "}",
    ]
    return "\n".join(lines) + "\n"


def _row_parameters(buffers: tuple[str, ...], simd: bool) -> tuple[str, ...]:
    bound = tuple(
        f"{_KINDS[name]} {name} [[buffer({index})]]"
        for index, name in enumerate(buffers)
    )
    return (*bound, *_ROW_THREADS, *(_SIMD_THREADS if simd else ()))


_RMSNORM_BUFFERS = ("y", "x", "w", "starts", "lengths")
_RMSNORM = _kernel(
    [
        "RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, one threadgroup per row.",
        "Each thread loads its 16 slots once into registers and adds their",
        "squares with fma; simd_sum and one threadgroup step give the sum.",
    ],
    RMSNORM_KERNEL,
    _row_parameters(_RMSNORM_BUFFERS, True),
    [
        f"threadgroup float partial[{_PARTIAL}];",
        *_ROW_BOUNDS,
        f"float v[{SLOTS}];",
        "float s = 0.0f;",
        f"for (uint i = 0u; i < {SLOTS}u; ++i) {{",
        "    uint c = lane + i * size;",
        "    v[i] = c < length ? x[start + c] : 0.0f;",
        "    s = fma(v[i], v[i], s);",
        "}",
        "s = simd_sum(s);",
        "if (simd_lane == 0u) {",
        "    partial[simd_group] = s;",
        "}",
        BARRIER,
        "float total = simd_sum(simd_lane < simd_groups ? partial[simd_lane] : 0.0f);",
        "float mean = precise::divide(total, float(length));",
        f"float inv = precise::rsqrt(mean + {EPS_TEXT});",
        f"for (uint i = 0u; i < {SLOTS}u; ++i) {{",
        "    uint c = lane + i * size;",
        "    if (c < length) {",
        "        y[start + c] = (v[i] * inv) * w[c];",
        "    }",
        "}",
    ],
)
_SWIGLU_BUFFERS = ("y", "a", "b", "starts", "lengths")
_SWIGLU = _kernel(
    [
        "SwiGLU: y = silu(a) * b = a / (1 + exp(-a)) * b in one pass, one",
        "threadgroup per row; no thread waits for another.",
    ],
    SWIGLU_KERNEL,
    _row_parameters(_SWIGLU_BUFFERS, False),
    [
        "uint start = starts[row];",
        "uint length = lengths[row];",
        "for (uint c = lane; c < length; c += size) {",
        "    float g = a[start + c];",
        "    float silu = precise::divide(g, 1.0f + precise::exp(-g));",
        "    y[start + c] = silu * b[start + c];",
        "}",
    ],
)
_CROSS_ENTROPY_BUFFERS = ("loss", "x", "targets", "starts", "lengths")
_CROSS_ENTROPY = _kernel(
    [
        "Cross-entropy: loss = (max - x[t]) + log(sum exp(x - max)) for each",
        "row. One pass keeps a running max m and a running sum d; when v > m",
        "the old sum is rescaled by exp(m - v). Each SIMD group merges its",
        "(m, d) pairs, one threadgroup step merges the SIMD groups, and lane 0",
        "writes one float. The softmax is never written.",
    ],
    CROSS_ENTROPY_KERNEL,
    _row_parameters(_CROSS_ENTROPY_BUFFERS, True),
    [
        f"threadgroup float max_partial[{_PARTIAL}];",
        f"threadgroup float sum_partial[{_PARTIAL}];",
        *_ROW_BOUNDS,
        f"float m = {LOWEST_TEXT};",
        "float d = 0.0f;",
        "for (uint c = lane; c < length; c += size) {",
        "    float v = x[start + c];",
        "    float scale = precise::exp(min(m, v) - max(m, v));",
        "    d = v > m ? fma(d, scale, 1.0f) : d + scale;",
        "    m = max(m, v);",
        "}",
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
        "if (lane == 0u) {",
        "    float margin = row_max - x[start + targets[row]];",
        "    loss[row] = margin + precise::log(total);",
        "}",
    ],
)


def _probe(name: str, function: str) -> str:
    return _kernel(
        [
            f"Probe: y = precise::{function}(x), one thread per argument, so a",
            "run can measure the error that the bounds assume.",
        ],
        name,
        (
            "device float* y [[buffer(0)]]",
            "device const float* x [[buffer(1)]]",
            "uint i [[thread_position_in_grid]]",
        ),
        [f"y[i] = precise::{function}(x[i]);"],
    )


KERNELS = {
    kernel.name: kernel
    for kernel in (
        LigerKernel(
            RMSNORM_KERNEL,
            RMSNORM,
            _RMSNORM_BUFFERS,
            _PARTIAL * FLOAT32_BYTES,
            1,
            "one read of x into registers; simd_sum and one threadgroup step",
            _RMSNORM,
        ),
        LigerKernel(
            SWIGLU_KERNEL,
            SWIGLU,
            _SWIGLU_BUFFERS,
            0,
            0,
            "one pass: read a and b, write y",
            _SWIGLU,
        ),
        LigerKernel(
            CROSS_ENTROPY_KERNEL,
            CROSS_ENTROPY,
            _CROSS_ENTROPY_BUFFERS,
            2 * _PARTIAL * FLOAT32_BYTES,
            1,
            "one online pass for max and sum; one float written for each row",
            _CROSS_ENTROPY,
        ),
        *(
            LigerKernel(
                name,
                None,
                ("y", "x"),
                0,
                0,
                f"precise::{function} at each argument",
                _probe(name, function),
            )
            for function, name in PROBES.items()
        ),
    )
}


def family_kernel(family: str) -> LigerKernel:
    """Return the fused kernel of one family."""

    if family not in FAMILIES:
        raise MetalLigerError(f"{family!r} is not a kernel family")
    return KERNELS[FAMILY_KERNELS[family]]


def liger_library() -> str:
    """Join the header and every kernel into the one text that compiles."""

    return HEADER + "".join("\n" + kernel.source for kernel in KERNELS.values())


def library_sha256() -> str:
    """Hash the exact library text that ``compile_shader`` receives."""

    return text_sha256(liger_library())
