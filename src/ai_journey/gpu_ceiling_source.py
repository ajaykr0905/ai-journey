"""Metal Shading Language source of the compute, bandwidth and bank kernels.

The Day 74 question "how far below peak" needs a peak, and a vendor table
describes no particular host. These kernels measure ceilings on the Apple GPU
that runs them instead, each one built to stress a single resource:

- ``fma_chains_<c>`` keeps ``c`` independent float32 chains in registers and
  applies ``x = fma(x, a, b)`` to every chain eight times per loop trip. The
  loop reads and writes no memory. Lane ``i % 32`` of every thread starts
  from row ``i % 32`` of a 32 x 32 table of seeded values, and the XOR of the
  final bit patterns is the thread's one output: it depends on every chain,
  so the compiler cannot drop a chain, and the CPU can rebuild it exactly.
- ``stream_copy`` and ``stream_read_reduce`` walk a buffer of float4 values
  with a grid stride, so one SIMD group touches 32 consecutive float4 values
  per step. The copy writes every value back; the read-reduce sums them into
  one float32 partial per thread.
- ``bank_rows_32`` and ``bank_rows_33`` read threadgroup memory: lane ``l``
  of each SIMD group reads logical word ``l * stride`` of a 32-column tile
  whose rows hold 32 words, or 33 with one word of padding. Each loop trip
  moves every lane down the same number of rows, which keeps the pattern
  across the lanes and leaves no load loop-invariant. Both kernels run the
  same instructions; only the row length differs, as an immediate.

Scalars are 32-bit ``constant uint&`` parameters, passed with ``arg_casts``
as in the Day 73 Metal lab, and every buffer binding is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .kernel_metal_geometry import KernelMetalError, require_integer


class GpuCeilingError(KernelMetalError):
    """Raised when a ceiling-lab control, launch, check or report is invalid."""


def require_count(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer by the Metal lab's rule, as this lab's error.

    ``require_integer`` refuses ``true`` and ``2.0`` with ``TypeError``; a
    value out of range becomes ``GpuCeilingError``.
    """

    try:
        return require_integer(value, name, minimum, maximum)
    except GpuCeilingError:
        raise
    except KernelMetalError as error:
        raise GpuCeilingError(str(error)) from error


FMA = "fma"
BANDWIDTH = "bandwidth"
BANK = "bank"
FAMILIES = (FMA, BANDWIDTH, BANK)

FMA_CHAIN_COUNTS = (1, 2, 4, 8, 16, 32)
FMA_STEPS_PER_TRIP = 8
# Thread i starts from row i % FMA_LANES of the start table; each row holds
# one value per chain, for up to FMA_START_COLUMNS chains.
FMA_LANES = 32
FMA_START_COLUMNS = max(FMA_CHAIN_COUNTS)
FMA_FLOPS_PER_STEP = 2

STREAM_COPY = "stream_copy"
STREAM_READ_REDUCE = "stream_read_reduce"
STREAM_KERNELS = (STREAM_COPY, STREAM_READ_REDUCE)
FLOAT4_BYTES = 16
FLOAT32_BYTES = 4

BANK_ROW_WORDS = (32, 33)
BANK_COLUMNS = 32
BANK_LANES = 32
BANK_MAX_STRIDE = 32
BANK_LOADS_PER_TRIP = 16
# Trip k moves every lane down k % BANK_SHIFT_PERIOD rows, so the deepest row
# read is 31 + 63 + 15, inside the tile's BANK_TILE_ROWS rows.
BANK_SHIFT_PERIOD = 64
BANK_TILE_ROWS = 112
BANK_WORD_MULTIPLIER = 2654435761
WORD_BYTES = 4


def fma_kernel_name(chains: int) -> str:
    """Name the compiled FMA kernel for one chain count."""

    if type(chains) is not int or chains not in FMA_CHAIN_COUNTS:
        raise GpuCeilingError(
            "chains must be one of " + ", ".join(map(str, FMA_CHAIN_COUNTS))
        )
    return f"fma_chains_{chains}"


def bank_kernel_name(row_words: int) -> str:
    """Name the compiled bank kernel for one row length in words."""

    if type(row_words) is not int or row_words not in BANK_ROW_WORDS:
        raise GpuCeilingError("row_words must be 32 or 33")
    return f"bank_rows_{row_words}"


_HEADER = """#include <metal_stdlib>
using namespace metal;

// One fused multiply-add on chain x. Eight steps of every chain per loop trip
// keep the loop branch a small share of the work.
#define FMA_STEP(x) x = fma(x, a, b);
#define EIGHT_TIMES(steps) steps steps steps steps steps steps steps steps
"""


def _signature(name: str, parameters: tuple[str, ...]) -> str:
    """Align each parameter under the first, as the Day 73 source does."""

    indent = "\n" + " " * (len(name) + 13)
    return f"kernel void {name}(" + ("," + indent).join(parameters) + ") {"


def _fma_kernel(chains: int) -> str:
    name = fma_kernel_name(chains)
    noun = "chain" if chains == 1 else "chains"
    lines = [
        "",
        f"// {chains} independent {noun} in registers; the loop touches no memory.",
        _signature(
            name,
            (
                "device uint* out [[buffer(0)]]",
                "device const float* start [[buffer(1)]]",
                "device const float* coefficients [[buffer(2)]]",
                "constant uint& n [[buffer(3)]]",
                "constant uint& trips [[buffer(4)]]",
                "uint i [[thread_position_in_grid]]",
            ),
        ),
        "    if (i < n) {",
        f"        const uint row = (i % {FMA_LANES}u) * {FMA_START_COLUMNS}u;",
        "        const float a = coefficients[0];",
        "        const float b = coefficients[1];",
        *(f"        float x{j} = start[row + {j}u];" for j in range(chains)),
        "        for (uint t = 0u; t < trips; ++t) {",
        "            EIGHT_TIMES(",
        *(
            "                "
            + " ".join(f"FMA_STEP(x{j})" for j in range(first, min(first + 4, chains)))
            for first in range(0, chains, 4)
        ),
        "            )",
        "        }",
        "        uint bits = as_type<uint>(x0);",
        *(f"        bits ^= as_type<uint>(x{j});" for j in range(1, chains)),
        "        out[i] = bits;",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


_STREAM = f"""
// Each thread copies every threads-th float4 from its own index, so one SIMD
// group reads and writes 32 consecutive float4 values per step.
kernel void {STREAM_COPY}(device float4* out [[buffer(0)]],
                        device const float4* in [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        constant uint& threads [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {{
    if (i < threads) {{
        for (uint e = i; e < n; e += threads) {{
            out[e] = in[e];
        }}
    }}
}}

// The same walk, summed into one partial per thread. The inputs are whole
// numbers below 1024, so every float32 sum is exact in any order.
kernel void {STREAM_READ_REDUCE}(device float* out [[buffer(0)]],
                               device const float4* in [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               constant uint& threads [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {{
    if (i < threads) {{
        float4 sum = float4(0.0f);
        for (uint e = i; e < n; e += threads) {{
            sum += in[e];
        }}
        out[i] = (sum.x + sum.y) + (sum.z + sum.w);
    }}
}}
"""


def _bank_kernel(row_words: int) -> str:
    name = bank_kernel_name(row_words)
    words = BANK_TILE_ROWS * row_words
    lines = [
        "",
        f"// Lane l reads logical word l * stride of a {BANK_COLUMNS}-column tile",
        f"// with {row_words}-word rows. Trip k moves every lane down "
        f"k % {BANK_SHIFT_PERIOD} rows",
        f"// and reads {BANK_LOADS_PER_TRIP} rows from there, so every load has "
        "the same bank pattern.",
        _signature(
            name,
            (
                "device uint* out [[buffer(0)]]",
                "constant uint& n [[buffer(1)]]",
                "constant uint& stride [[buffer(2)]]",
                "constant uint& trips [[buffer(3)]]",
                "uint i [[thread_position_in_grid]]",
                "uint t [[thread_index_in_threadgroup]]",
                "uint group [[threads_per_threadgroup]]",
            ),
        ),
        f"    threadgroup uint tile[{words}];",
        f"    for (uint w = t; w < {words}u; w += group) {{",
        f"        tile[w] = w * {BANK_WORD_MULTIPLIER}u;",
        "    }",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"    const uint logical = (t % {BANK_LANES}u) * stride;",
        f"    const uint first = (logical / {BANK_COLUMNS}u) * {row_words}u"
        f" + logical % {BANK_COLUMNS}u;",
        "    uint s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;",
        "    for (uint k = 0u; k < trips; ++k) {",
        f"        const uint q = first + (k % {BANK_SHIFT_PERIOD}u) * {row_words}u;",
        *(
            f"        s{load % 4} += tile[q + {load * row_words}u];"
            for load in range(BANK_LOADS_PER_TRIP)
        ),
        "    }",
        "    if (i < n) {",
        "        out[i] = (s0 + s1) + (s2 + s3);",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


SHADER_SOURCE = (
    _HEADER
    + "".join(_fma_kernel(chains) for chains in FMA_CHAIN_COUNTS)
    + _STREAM
    + "".join(_bank_kernel(row_words) for row_words in BANK_ROW_WORDS)
)


@dataclass(frozen=True)
class CeilingKernel:
    """One compiled kernel: its family, buffer bindings and scalar bindings."""

    name: str
    family: str
    buffers: tuple[str, ...]
    scalar_buffers: tuple[int, ...]

    @property
    def arg_casts(self) -> dict[int, str]:
        """Pass each scalar as 32 bits, the width of ``constant uint&``."""

        return {index: "int32" for index in self.scalar_buffers}

    def to_mapping(self) -> dict[str, object]:
        """Return the bindings a report records."""

        return {
            "name": self.name,
            "family": self.family,
            "buffers": list(self.buffers),
            "scalar_buffers": list(self.scalar_buffers),
        }


KERNELS = {
    **{
        fma_kernel_name(chains): CeilingKernel(
            name=fma_kernel_name(chains),
            family=FMA,
            buffers=("out", "start", "coefficients", "n", "trips"),
            scalar_buffers=(3, 4),
        )
        for chains in FMA_CHAIN_COUNTS
    },
    **{
        name: CeilingKernel(
            name=name,
            family=BANDWIDTH,
            buffers=("out", "in", "n", "threads"),
            scalar_buffers=(2, 3),
        )
        for name in STREAM_KERNELS
    },
    **{
        bank_kernel_name(row_words): CeilingKernel(
            name=bank_kernel_name(row_words),
            family=BANK,
            buffers=("out", "n", "stride", "trips"),
            scalar_buffers=(1, 2, 3),
        )
        for row_words in BANK_ROW_WORDS
    },
}


def bank_tile_bytes(row_words: int) -> int:
    """Threadgroup memory one bank kernel declares, in bytes."""

    bank_kernel_name(row_words)
    return BANK_TILE_ROWS * row_words * WORD_BYTES


def source_sha256() -> str:
    """Hash the exact source text that ``compile_shader`` receives."""

    return sha256(SHADER_SOURCE.encode("ascii")).hexdigest()
