"""Metal Shading Language source of the Day 82 matmul kernels, with and without MMA.

Curriculum Day 82 follows GPU MODE Lecture 23, "Tensor Cores", and Lecture
15, "CUTLASS". Apple GPUs have no NVIDIA Tensor Cores. Their matrix path is
``simdgroup_matrix``: the 32 threads of one SIMD group hold one 8 x 8
matrix together, and ``simdgroup_multiply_accumulate`` multiplies two of
them into a third. This lab compares a matmul with that path and without
it, on the same tiling. ``C[m, n] = A[m, k] @ B[k, n]`` is row-major, C is
float32, and every kernel adds its products into float32 accumulators.

- S1 (``tc_s1_float``): float32 inputs, scalar ``fma`` only. Each of the 128
  threads of a group owns a 2 x 4 block of float accumulators, at the rows
  ``tr + 16 i`` and the columns ``tc + 8 j`` of the group tile.
- M1, M2 and M3 (``tc_m1_float``, ``tc_m2_half``, ``tc_m3_bfloat``): inputs
  of float, half and bfloat. Each of the four SIMD groups owns a 16 x 16
  block of the group tile as 2 x 2 accumulators of ``simdgroup_float8x8``,
  and multiplies 8 x 8 fragments of the input type into them.

All four kernels share the tiling. One group of 128 threads owns one 32 x 32
tile of C and walks K in steps of 16. In each step the group copies a 32 x
16 tile of A and a 16 x 32 tile of B into threadgroup memory, zero-filled
past the matrices. Each row of a tile holds 4 more slots of padding, so the
rows of a fragment start in different banks. Each thread holds 8 float
accumulators in S1 and in M1 to M3.

These rules keep each kernel correct and its declared bound valid:

- ``threadgroup_barrier`` appears only in the K loop, whose bound is the
  same for every thread of the group, so every thread reaches every
  barrier. ``simdgroup_barrier`` appears only in the edge store of an MMA
  kernel, in a branch that depends on the fragment position alone, so
  every lane of the SIMD group takes it. ``require_uniform_barriers``
  checks both rules on each text.
- A slot of a tile past M, N or K holds 0, and a product with a 0 adds
  nothing. A store past M or N never happens: an MMA fragment that crosses
  an edge goes through a staging slot in threadgroup memory, and each lane
  stores only its elements inside C.

Each kernel has a SHA-256 digest of its exact text. The bfloat kernel
compiles in a library of its own, so a host whose compiler refuses
``simdgroup_bfloat8x8`` still runs the other kernels and records M3 as
unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .matmul_tiled_source import BARRIER
from .matmul_tiled_variants import text_sha256
from .metal_flash_source import MetalFlashError
from .metal_flash_source import require_size as _flash_size

HEADER = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
SIMD_BARRIER = "simdgroup_barrier(mem_flags::mem_threadgroup);"
TILE = 32
STEP = 16
FRAGMENT = 8
GROUP = 128
SIMD_WIDTH = 32
SIMD_GROUPS = GROUP // SIMD_WIDTH
WARP_TILE = TILE // 2
FRAGMENTS = WARP_TILE // FRAGMENT
PAD = 4
A_STRIDE = STEP + PAD
B_STRIDE = TILE + PAD
A_SLOTS = TILE * A_STRIDE
B_SLOTS = STEP * B_STRIDE
STAGE_FLOATS = SIMD_GROUPS * FRAGMENT * FRAGMENT
FLOAT32_BYTES = 4
ROW_THREADS = 16
COLUMN_THREADS = 8
THREAD_ROWS = TILE // ROW_THREADS
THREAD_COLUMNS = TILE // COLUMN_THREADS
MAX_DIMENSION = 8192

S1 = "s1"
M1 = "m1"
M2 = "m2"
M3 = "m3"
VARIANTS = (S1, M1, M2, M3)
FLOAT32 = "float32"
FLOAT16 = "float16"
BFLOAT16 = "bfloat16"
PRECISIONS = (FLOAT32, FLOAT16, BFLOAT16)
PRECISION = {S1: FLOAT32, M1: FLOAT32, M2: FLOAT16, M3: BFLOAT16}
METAL_TYPE = {FLOAT32: "float", FLOAT16: "half", BFLOAT16: "bfloat"}
INPUT_BYTES = {FLOAT32: 4, FLOAT16: 2, BFLOAT16: 2}
KERNEL_NAME = {
    S1: "tc_s1_float",
    M1: "tc_m1_float",
    M2: "tc_m2_half",
    M3: "tc_m3_bfloat",
}
MMA = {S1: False, M1: True, M2: True, M3: True}
PROBE_VERSION = "tc_probe_version"
MAIN_LIBRARY = "main"
BFLOAT_LIBRARY = "bfloat"
LIBRARIES = (MAIN_LIBRARY, BFLOAT_LIBRARY)
K_LOOP = f"for (uint k0 = 0u; k0 < k; k0 += {STEP}u) {{"
EDGE_BRANCH = "} else {"
SCALARS = {3: "m", 4: "n", 5: "k"}
ARG_CASTS = {index: "int32" for index in SCALARS}
_NAME = re.compile(r"tc_[a-z0-9_]{1,40}")


class MetalTcError(MetalFlashError):
    """Raised when a Day 82 kernel, control, launch, check or report is invalid."""


def require_size(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer as the Metal labs do, and raise this lab's error."""

    try:
        return _flash_size(value, name, minimum, maximum)
    except MetalTcError:
        raise
    except MetalFlashError as error:
        raise MetalTcError(str(error)) from error


def require_uniform_barriers(text: str) -> tuple[int, int]:
    """Check where each barrier sits; return the threadgroup and SIMD counts.

    The generated text puts each opening brace at the end of a line and each
    closing brace at the start of one. A ``threadgroup_barrier`` must sit
    directly in the K loop. A ``simdgroup_barrier`` must sit in the edge
    branch of the store, whose condition reads only the fragment position.
    """

    blocks: list[str] = []
    counts = [0, 0]
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("}"):
            if not blocks:
                raise MetalTcError("a closing brace has no open block")
            blocks.pop()
        if line == BARRIER:
            counts[0] += 1
            if blocks[1:] != [K_LOOP]:
                raise MetalTcError("a threadgroup barrier sits outside the K loop")
        if line == SIMD_BARRIER:
            counts[1] += 1
            if not blocks or blocks[-1] != EDGE_BRANCH or K_LOOP in blocks:
                raise MetalTcError("a SIMD barrier sits outside the edge store")
        if line.endswith("{"):
            blocks.append(line)
    if blocks:
        raise MetalTcError("a block of the kernel text is not closed")
    return counts[0], counts[1]


def threadgroup_bytes(variant: str) -> int:
    """Bytes of the two padded tiles, plus the staging slots of an MMA kernel."""

    size = INPUT_BYTES[PRECISION[variant]]
    stage = STAGE_FLOATS * FLOAT32_BYTES if MMA[variant] else 0
    return (A_SLOTS + B_SLOTS) * size + stage


@dataclass(frozen=True)
class TcKernel:
    """One kernel of the lab: its variant, input type, library, memory and text."""

    name: str
    variant: str | None
    library: str
    group_size: int
    threadgroup_bytes: int
    barriers: tuple[int, int]
    summary: str
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalTcError("a kernel name must be tc_ and a short slug")
        if self.variant is None:
            if self.name != PROBE_VERSION or self.library != MAIN_LIBRARY:
                raise MetalTcError("only the version probe has no variant")
        elif self.name != KERNEL_NAME.get(self.variant):
            raise MetalTcError(f"{self.name} must be the kernel of its variant")
        expected = BFLOAT_LIBRARY if self.variant == M3 else MAIN_LIBRARY
        if self.library != expected:
            raise MetalTcError(f"{self.name} belongs to the {expected} library")
        require_size(self.group_size, "group_size", 1, 1024)
        require_size(self.threadgroup_bytes, "threadgroup_bytes", 0, 32 * 1024)
        if not isinstance(self.summary, str) or not self.summary.isascii():
            raise MetalTcError(f"{self.name} summary must be ASCII text")
        if not isinstance(self.source, str) or not self.source.isascii():
            raise MetalTcError(f"{self.name} text must be ASCII")
        if f"kernel void {self.name}(" not in self.source:
            raise MetalTcError(f"{self.name} text does not define the kernel")
        if require_uniform_barriers(self.source) != self.barriers:
            raise MetalTcError(f"{self.name} declares the wrong barrier counts")

    @property
    def precision(self) -> str | None:
        """The input type of the kernel; the probe has none."""

        return None if self.variant is None else PRECISION[self.variant]

    @property
    def source_sha256(self) -> str:
        """Hash the exact text of this kernel."""

        return text_sha256(self.source)

    def to_mapping(self) -> dict[str, object]:
        """Return the kernel as a report records it."""

        return {
            "name": self.name,
            "variant": self.variant,
            "precision": self.precision,
            "library": self.library,
            "group_size": self.group_size,
            "threadgroup_bytes": self.threadgroup_bytes,
            "threadgroup_barriers": self.barriers[0],
            "simdgroup_barriers": self.barriers[1],
            "summary": self.summary,
            "source_sha256": self.source_sha256,
        }


def _kernel(comment: list[str], name: str, kind: str, extra: tuple, body) -> str:
    head = f"kernel void {name}("
    separator = ",\n" + " " * len(head)
    parameters = (
        "device float* c [[buffer(0)]]",
        f"device const {kind}* a [[buffer(1)]]",
        f"device const {kind}* b [[buffer(2)]]",
        "constant uint& m [[buffer(3)]]",
        "constant uint& n [[buffer(4)]]",
        "constant uint& k [[buffer(5)]]",
        "uint group [[threadgroup_position_in_grid]]",
        "uint lane [[thread_position_in_threadgroup]]",
        *extra,
    )
    lines = [
        *(f"// {line}" for line in comment),
        *(head + separator.join(parameters) + ") {").splitlines(),
        *(f"    {line}" for line in body),
        "}",
    ]
    return "\n".join(lines) + "\n"


def _origin() -> list[str]:
    return [
        f"uint tiles_n = (n + {TILE - 1}u) / {TILE}u;",
        f"uint row0 = (group / tiles_n) * {TILE}u;",
        f"uint col0 = (group % tiles_n) * {TILE}u;",
    ]


def _stage_tiles(kind: str) -> list[str]:
    """Copy one 32 x 16 tile of A and one 16 x 32 tile of B, zero past the edges."""

    zero = f"{kind}(0.0f)"
    return [
        f"    for (uint e = lane; e < {TILE * STEP}u; e += {GROUP}u) {{",
        f"        uint ar = row0 + e / {STEP}u;",
        f"        uint ak = k0 + e % {STEP}u;",
        f"        a_tile[(e / {STEP}u) * {A_STRIDE}u + e % {STEP}u] =",
        f"            ar < m && ak < k ? a[ar * k + ak] : {zero};",
        f"        uint bk = k0 + e / {TILE}u;",
        f"        uint bc = col0 + e % {TILE}u;",
        f"        b_tile[(e / {TILE}u) * {B_STRIDE}u + e % {TILE}u] =",
        f"            bk < k && bc < n ? b[bk * n + bc] : {zero};",
        "    }",
        f"    {BARRIER}",
    ]


def _tiles(kind: str) -> list[str]:
    return [
        f"threadgroup {kind} a_tile[{A_SLOTS}];",
        f"threadgroup {kind} b_tile[{B_SLOTS}];",
    ]


def _scalar() -> str:
    rows, columns = THREAD_ROWS, THREAD_COLUMNS
    across, down = COLUMN_THREADS, ROW_THREADS
    return _kernel(
        [
            "S1: one group of 128 threads per 32 x 32 tile of C, float32 inputs.",
            "Each thread owns the rows tr + 16 i and the columns tc + 8 j of the",
            "tile and adds each product with a scalar fma into its own float.",
        ],
        KERNEL_NAME[S1],
        "float",
        (),
        [
            *_tiles("float"),
            *_origin(),
            f"uint tr = lane / {across}u;",
            f"uint tc = lane % {across}u;",
            f"float acc[{rows}][{columns}];",
            f"for (uint i = 0u; i < {rows}u; ++i) {{",
            f"    for (uint j = 0u; j < {columns}u; ++j) {{",
            "        acc[i][j] = 0.0f;",
            "    }",
            "}",
            K_LOOP,
            *_stage_tiles("float"),
            f"    for (uint q = 0u; q < {STEP}u; ++q) {{",
            f"        float av[{rows}];",
            f"        for (uint i = 0u; i < {rows}u; ++i) {{",
            f"            av[i] = a_tile[(tr + {down}u * i) * {A_STRIDE}u + q];",
            "        }",
            f"        for (uint j = 0u; j < {columns}u; ++j) {{",
            f"            float bv = b_tile[q * {B_STRIDE}u + tc + {across}u * j];",
            f"            for (uint i = 0u; i < {rows}u; ++i) {{",
            "                acc[i][j] = fma(av[i], bv, acc[i][j]);",
            "            }",
            "        }",
            "    }",
            f"    {BARRIER}",
            "}",
            f"for (uint i = 0u; i < {rows}u; ++i) {{",
            f"    uint row = row0 + tr + {down}u * i;",
            f"    for (uint j = 0u; j < {columns}u; ++j) {{",
            f"        uint col = col0 + tc + {across}u * j;",
            "        if (row < m && col < n) {",
            "            c[row * n + col] = acc[i][j];",
            "        }",
            "    }",
            "}",
        ],
    )


def _matrix(variant: str) -> str:
    kind = METAL_TYPE[PRECISION[variant]]
    f, w, d = FRAGMENTS, WARP_TILE, FRAGMENT
    return _kernel(
        [
            f"{variant.upper()}: the tiling of S1 with {kind} inputs on the simdgroup "
            "matrix path.",
            "Each SIMD group owns a 16 x 16 block as 2 x 2 simdgroup_float8x8",
            f"accumulators and multiplies 8 x 8 {kind} fragments of the tiles into",
            "them. A fragment inside C is stored directly, and a fragment that",
            "crosses an edge goes through its staging slot.",
        ],
        KERNEL_NAME[variant],
        kind,
        (
            "uint simd_lane [[thread_index_in_simdgroup]]",
            "uint simd_group [[simdgroup_index_in_threadgroup]]",
        ),
        [
            *_tiles(kind),
            f"threadgroup float stage[{STAGE_FLOATS}];",
            *_origin(),
            f"uint sr = (simd_group / 2u) * {w}u;",
            f"uint sc = (simd_group % 2u) * {w}u;",
            f"simdgroup_float8x8 acc[{f}][{f}];",
            f"for (uint i = 0u; i < {f}u; ++i) {{",
            f"    for (uint j = 0u; j < {f}u; ++j) {{",
            "        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);",
            "    }",
            "}",
            K_LOOP,
            *_stage_tiles(kind),
            f"    for (uint q = 0u; q < {STEP}u; q += {d}u) {{",
            f"        simdgroup_matrix<{kind}, 8, 8> af[{f}];",
            f"        simdgroup_matrix<{kind}, 8, 8> bf[{f}];",
            f"        for (uint i = 0u; i < {f}u; ++i) {{",
            f"            simdgroup_load(af[i], a_tile + (sr + {d}u * i) * {A_STRIDE}u "
            f"+ q, {A_STRIDE});",
            "        }",
            f"        for (uint j = 0u; j < {f}u; ++j) {{",
            f"            simdgroup_load(bf[j], b_tile + q * {B_STRIDE}u + sc + {d}u * j, "
            f"{B_STRIDE});",
            "        }",
            f"        for (uint i = 0u; i < {f}u; ++i) {{",
            f"            for (uint j = 0u; j < {f}u; ++j) {{",
            "                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], "
            "acc[i][j]);",
            "            }",
            "        }",
            "    }",
            f"    {BARRIER}",
            "}",
            f"for (uint i = 0u; i < {f}u; ++i) {{",
            f"    for (uint j = 0u; j < {f}u; ++j) {{",
            f"        uint row = row0 + sr + {d}u * i;",
            f"        uint col = col0 + sc + {d}u * j;",
            f"        if (row + {d}u <= m && col + {d}u <= n) {{",
            "            simdgroup_store(acc[i][j], c + row * n + col, n);",
            f"        {EDGE_BRANCH}",
            f"            simdgroup_store(acc[i][j], stage + simd_group * {d * d}u, {d});",
            f"            {SIMD_BARRIER}",
            f"            for (uint e = simd_lane; e < {d * d}u; e += {SIMD_WIDTH}u) {{",
            f"                uint r = row + e / {d}u;",
            f"                uint s = col + e % {d}u;",
            "                if (r < m && s < n) {",
            f"                    c[r * n + s] = stage[simd_group * {d * d}u + e];",
            "                }",
            "            }",
            f"            {SIMD_BARRIER}",
            "        }",
            "    }",
            "}",
        ],
    )


_PROBE = (
    "\n".join(
        [
            "// Probe: write the Metal language version that compiled this library.",
            f"kernel void {PROBE_VERSION}(device uint* out [[buffer(0)]],",
            " " * len(f"kernel void {PROBE_VERSION}(")
            + "uint i [[thread_position_in_grid]]) {",
            "    if (i == 0u) {",
            "        out[0] = __METAL_VERSION__;",
            "    }",
            "}",
        ]
    )
    + "\n"
)


def _registry() -> dict[str, TcKernel]:
    kernels = [
        TcKernel(
            KERNEL_NAME[S1],
            S1,
            MAIN_LIBRARY,
            GROUP,
            threadgroup_bytes(S1),
            (2, 0),
            "scalar fma from the padded tiles; 8 float accumulators per thread",
            _scalar(),
        )
    ]
    for variant in (M1, M2, M3):
        kernels.append(
            TcKernel(
                KERNEL_NAME[variant],
                variant,
                BFLOAT_LIBRARY if variant == M3 else MAIN_LIBRARY,
                GROUP,
                threadgroup_bytes(variant),
                (2, 2),
                f"simdgroup_multiply_accumulate on {METAL_TYPE[PRECISION[variant]]} "
                "8 x 8 fragments into 4 float accumulators per SIMD group",
                _matrix(variant),
            )
        )
    kernels.append(
        TcKernel(
            PROBE_VERSION,
            None,
            MAIN_LIBRARY,
            SIMD_WIDTH,
            0,
            (0, 0),
            "writes __METAL_VERSION__ once",
            _PROBE,
        )
    )
    return {kernel.name: kernel for kernel in kernels}


KERNELS = _registry()


def library_source(library: str) -> str:
    """Join the header and the kernels of one library into the text that compiles."""

    if library not in LIBRARIES:
        raise MetalTcError(f"{library!r} is not a library of this lab")
    return HEADER + "".join(
        "\n" + kernel.source for kernel in KERNELS.values() if kernel.library == library
    )


def library_sha256(library: str) -> str:
    """Hash the exact text of one library that ``compile_shader`` receives."""

    return text_sha256(library_source(library))
