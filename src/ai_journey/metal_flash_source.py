"""Metal Shading Language source of the Day 81 attention forward kernels.

Curriculum Day 81 (Track B) follows GPU MODE Lecture 12, "Flash
Attention". Triton has no macOS build, so this lab writes attention in
Metal and runs it on the Apple GPU through ``torch.mps.compile_shader``.
Each sequence has its own length, and each row of Q, K, V and O holds
``HEAD_DIM = 64`` floats. The scale is ``1 / sqrt(64) = 0.125``, a power of
two, so the scale product is exact for normal floats.

- N0, naive attention, three kernels. ``flash_n0_scores`` writes
  ``S = 0.125 Q K^T`` into an N x N device buffer for each sequence.
  ``flash_n0_softmax`` turns each row of S into ``P = softmax(S)`` in place.
  ``flash_n0_pv`` writes ``O = P V``. The N x N buffer goes to device
  memory and comes back twice.
- F1, flash-attention forward, one kernel. One threadgroup takes one block
  of 32 queries. It loads K and V in tiles of 32 rows into threadgroup
  memory, keeps a running max and a running sum for each query, and
  rescales the sum and the output when the max grows. S and P never go to
  device memory.

Each kernel has a causal variant, in which query ``i`` sees the keys
``j <= i`` only. These rules keep each kernel correct and its declared
error bound valid:

- Every barrier is in the kernel body or in the K/V tile loop. The bound of
  that loop comes from the sequence length and the query block only, so it
  is the same for every thread of the group, and every thread reaches every
  barrier. ``require_uniform_barriers`` checks this rule on each text.
- A query row at or past the sequence length writes nothing, and a key row
  at or past the length is never read.
- ``precise::exp`` and ``precise::divide`` fix the math whatever the compile
  options are. A probe kernel lets a run measure the error of
  ``precise::exp`` that the bound assumes.

Each kernel has a SHA-256 digest of its exact text, and the library joins
every kernel under one header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .matmul_tiled_source import BARRIER, HEADER
from .matmul_tiled_variants import text_sha256
from .metal_fused_source import LOWEST_TEXT, MetalFusedError, require_count

FLOAT32_BYTES = 4
UINT32_BYTES = 4
HEAD_DIM = 64
SCALE = 0.125
SCALE_TEXT = "0.125f"
BLOCK_QUERIES = 32
BLOCK_KEYS = 32
PARTS = 4
FLASH_GROUP = BLOCK_QUERIES * PARTS
NAIVE_GROUP = 256
PV_GROUP = HEAD_DIM
PROBE_GROUP = 256
SIMD_WIDTH = 32
TILE_FLOATS = BLOCK_KEYS * HEAD_DIM
DIMS_PER_PART = HEAD_DIM // PARTS

N0 = "n0"
F1 = "f1"
VARIANTS = (N0, F1)
FULL = "full"
CAUSAL = "causal"
MODES = (FULL, CAUSAL)
PROBE_EXP = "flash_probe_exp"
TILE_LOOP = f"for (uint tile = 0u; tile < end; tile += {BLOCK_KEYS}u) {{"
_NAME = re.compile(r"flash_[a-z0-9_]{1,40}")


class MetalFlashError(MetalFusedError):
    """Raised when a Day 81 kernel, control, launch, check or report is invalid."""


def require_size(value: object, name: str, minimum: int, maximum: int) -> int:
    """Bound an integer as the Metal labs do, and raise this lab's error."""

    try:
        return require_count(value, name, minimum, maximum)
    except MetalFlashError:
        raise
    except MetalFusedError as error:
        raise MetalFlashError(str(error)) from error


def require_uniform_barriers(text: str) -> int:
    """Check that every barrier sits in the body or in the K/V tile loop.

    The generated text puts each opening brace at the end of a line and each
    closing brace at the start of one. The tile loop takes its bound ``end``
    from the sequence and the query block, which every thread of a group
    shares. Return the count of barriers in the text.
    """

    blocks: list[str] = []
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("}"):
            if not blocks:
                raise MetalFlashError("a closing brace has no open block")
            blocks.pop()
        if line == BARRIER:
            count += 1
            if not blocks or not all(opener == TILE_LOOP for opener in blocks[1:]):
                raise MetalFlashError(
                    "a barrier sits inside a block that depends on the thread"
                )
        if line.endswith("{"):
            blocks.append(line)
    if blocks:
        raise MetalFlashError("a block of the kernel text is not closed")
    return count


def kernel_names(variant: str, mode: str) -> tuple[str, ...]:
    """Return the kernels of one variant and mode, in launch order."""

    if variant not in VARIANTS:
        raise MetalFlashError(f"{variant!r} is not an attention variant")
    if mode not in MODES:
        raise MetalFlashError(f"{mode!r} is not an attention mode")
    suffix = "_causal" if mode == CAUSAL else ""
    if variant == N0:
        return tuple(f"flash_n0_{step}{suffix}" for step in ("scores", "softmax", "pv"))
    return (f"flash_f1{suffix}",)


@dataclass(frozen=True)
class FlashKernel:
    """One kernel of the library: its role, buffers, group, memory and text."""

    name: str
    variant: str | None
    mode: str | None
    buffers: tuple[str, ...]
    group_size: int
    threadgroup_bytes: int
    barriers: int
    summary: str
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalFlashError("a kernel name must be flash_ and a short slug")
        if self.variant is None:
            if self.mode is not None or self.name != PROBE_EXP:
                raise MetalFlashError("only the probe has no variant")
        elif self.name not in kernel_names(self.variant, self.mode):
            raise MetalFlashError(f"{self.name} must belong to its variant and mode")
        if (
            type(self.buffers) is not tuple
            or not self.buffers
            or not all(isinstance(name, str) for name in self.buffers)
        ):
            raise MetalFlashError(f"{self.name} must name its buffers")
        require_size(self.group_size, "group_size", SIMD_WIDTH, 1024)
        require_size(self.threadgroup_bytes, "threadgroup_bytes", 0, 32 * 1024)
        if not isinstance(self.summary, str) or not self.summary.isascii():
            raise MetalFlashError(f"{self.name} summary must be ASCII text")
        if f"kernel void {self.name}(" not in self.source:
            raise MetalFlashError(f"{self.name} text does not define the kernel")
        if require_uniform_barriers(self.source) != self.barriers:
            raise MetalFlashError(f"{self.name} declares the wrong barrier count")

    @property
    def source_sha256(self) -> str:
        """Hash the exact text of this kernel."""

        return text_sha256(self.source)

    def to_mapping(self) -> dict[str, object]:
        """Return the kernel as a report records it."""

        return {
            "name": self.name,
            "variant": self.variant,
            "mode": self.mode,
            "buffers": list(self.buffers),
            "group_size": self.group_size,
            "threadgroup_bytes": self.threadgroup_bytes,
            "barriers": self.barriers,
            "summary": self.summary,
            "source_sha256": self.source_sha256,
        }


_KINDS = {
    "o": "device float*",
    "s": "device float*",
    "y": "device float*",
    "q": "device const float*",
    "k": "device const float*",
    "v": "device const float*",
    "x": "device const float*",
    "starts": "device const uint*",
    "lengths": "device const uint*",
    "score_starts": "device const uint*",
    "seqs": "device const uint*",
    "rows": "device const uint*",
    "blocks": "device const uint*",
}
_GROUP_THREADS = (
    "uint group [[threadgroup_position_in_grid]]",
    "uint lane [[thread_position_in_threadgroup]]",
)
_SIMD_THREADS = (
    "uint simd_lane [[thread_index_in_simdgroup]]",
    "uint simd_group [[simdgroup_index_in_threadgroup]]",
)


def _kernel(comment, name: str, buffers, threads, body) -> str:
    head = f"kernel void {name}("
    separator = ",\n" + " " * len(head)
    parameters = (
        *(
            f"{_KINDS[buffer]} {buffer} [[buffer({index})]]"
            for index, buffer in enumerate(buffers)
        ),
        *threads,
    )
    lines = [
        *(f"// {line}" for line in comment),
        *(head + separator.join(parameters) + ") {").splitlines(),
        *(f"    {line}" for line in body),
        "}",
    ]
    return "\n".join(lines) + "\n"


def _query_row() -> list[str]:
    return [
        "uint seq = seqs[group];",
        "uint start = starts[seq];",
        "uint length = lengths[seq];",
        "uint row = rows[group];",
        "uint base = score_starts[seq] + row * length;",
    ]


def _limit(mode: str) -> str:
    return "uint limit = row + 1u;" if mode == CAUSAL else "uint limit = length;"


_SCORES_BUFFERS = ("s", "q", "k", "starts", "lengths", "score_starts", "seqs", "rows")
_SOFTMAX_BUFFERS = ("s", "starts", "lengths", "score_starts", "seqs", "rows")
_PV_BUFFERS = ("o", "s", "v", "starts", "lengths", "score_starts", "seqs", "rows")
_FLASH_BUFFERS = ("o", "q", "k", "v", "starts", "lengths", "seqs", "blocks")
_PARTIALS = NAIVE_GROUP // SIMD_WIDTH


def _scores(mode: str) -> str:
    return _kernel(
        [
            "N0 step 1: S[row][j] = 0.125 * dot(q[row], k[j]) for each key j of the",
            "row, one threadgroup per query row; S goes to device memory.",
        ],
        kernel_names(N0, mode)[0],
        _SCORES_BUFFERS,
        _GROUP_THREADS + ("uint size [[threads_per_threadgroup]]",),
        [
            f"threadgroup float q_row[{HEAD_DIM}];",
            *_query_row(),
            _limit(mode),
            f"if (lane < {HEAD_DIM}u) {{",
            f"    q_row[lane] = q[(start + row) * {HEAD_DIM}u + lane];",
            "}",
            BARRIER,
            "for (uint j = lane; j < limit; j += size) {",
            f"    uint key = (start + j) * {HEAD_DIM}u;",
            "    float dot = 0.0f;",
            f"    for (uint c = 0u; c < {HEAD_DIM}u; ++c) {{",
            "        dot = fma(q_row[c], k[key + c], dot);",
            "    }",
            f"    s[base + j] = dot * {SCALE_TEXT};",
            "}",
        ],
    )


def _softmax(mode: str) -> str:
    return _kernel(
        [
            "N0 step 2: P = softmax(S) in place for each row: simd_max and one",
            "threadgroup step give the max, then exp, then simd_sum and one step",
            "give the sum, then each value is divided by the sum.",
        ],
        kernel_names(N0, mode)[1],
        _SOFTMAX_BUFFERS,
        _GROUP_THREADS + ("uint size [[threads_per_threadgroup]]",) + _SIMD_THREADS,
        [
            f"threadgroup float max_partial[{_PARTIALS}];",
            f"threadgroup float sum_partial[{_PARTIALS}];",
            *_query_row(),
            _limit(mode),
            f"float m = {LOWEST_TEXT};",
            "for (uint j = lane; j < limit; j += size) {",
            "    m = max(m, s[base + j]);",
            "}",
            "m = simd_max(m);",
            "if (simd_lane == 0u) {",
            "    max_partial[simd_group] = m;",
            "}",
            BARRIER,
            f"float top = simd_max(simd_lane < {_PARTIALS}u ? max_partial[simd_lane] "
            f": {LOWEST_TEXT});",
            "float d = 0.0f;",
            "for (uint j = lane; j < limit; j += size) {",
            "    float e = precise::exp(s[base + j] - top);",
            "    s[base + j] = e;",
            "    d += e;",
            "}",
            "d = simd_sum(d);",
            "if (simd_lane == 0u) {",
            "    sum_partial[simd_group] = d;",
            "}",
            BARRIER,
            f"float total = simd_sum(simd_lane < {_PARTIALS}u ? sum_partial[simd_lane] "
            ": 0.0f);",
            "for (uint j = lane; j < limit; j += size) {",
            "    s[base + j] = precise::divide(s[base + j], total);",
            "}",
        ],
    )


def _pv(mode: str) -> str:
    return _kernel(
        [
            "N0 step 3: O[row] = sum_j P[row][j] * v[j], one thread per column of",
            "the head and one threadgroup per query row.",
        ],
        kernel_names(N0, mode)[2],
        _PV_BUFFERS,
        _GROUP_THREADS,
        [
            *_query_row(),
            _limit(mode),
            "float acc = 0.0f;",
            "for (uint j = 0u; j < limit; ++j) {",
            f"    acc = fma(s[base + j], v[(start + j) * {HEAD_DIM}u + lane], acc);",
            "}",
            f"o[(start + row) * {HEAD_DIM}u + lane] = acc;",
        ],
    )


def _flash(mode: str) -> str:
    causal = mode == CAUSAL
    end = f"min(length, first + {BLOCK_QUERIES}u)" if causal else "length"
    valid = "key < length && key <= row" if causal else "key < length"
    d, p = HEAD_DIM, PARTS
    return _kernel(
        [
            "F1: flash-attention forward, one threadgroup per block of 32 queries.",
            "Four threads share one query row and own the columns part + 4 t.",
            "Each pass loads one tile of 32 K rows and 32 V rows into threadgroup",
            "memory, computes 32 scores, and updates the running max m, the running",
            "sum l and the output acc with the online rescaling. S and P stay in",
            "registers.",
        ],
        kernel_names(F1, mode)[0],
        _FLASH_BUFFERS,
        _GROUP_THREADS,
        [
            f"threadgroup float k_tile[{TILE_FLOATS}];",
            f"threadgroup float v_tile[{TILE_FLOATS}];",
            "uint seq = seqs[group];",
            "uint start = starts[seq];",
            "uint length = lengths[seq];",
            f"uint first = blocks[group] * {BLOCK_QUERIES}u;",
            f"uint row = first + lane / {p}u;",
            f"uint part = lane % {p}u;",
            "bool live = row < length;",
            f"float qv[{DIMS_PER_PART}];",
            f"float acc[{DIMS_PER_PART}];",
            f"for (uint t = 0u; t < {DIMS_PER_PART}u; ++t) {{",
            "    qv[t] = 0.0f;",
            "    if (live) {",
            f"        qv[t] = q[(start + row) * {d}u + t * {p}u + part];",
            "    }",
            "    acc[t] = 0.0f;",
            "}",
            f"float m = {LOWEST_TEXT};",
            "float l = 0.0f;",
            f"uint end = {end};",
            TILE_LOOP,
            f"    for (uint e = lane; e < {TILE_FLOATS}u; e += {FLASH_GROUP}u) {{",
            f"        uint key = tile + e / {d}u;",
            "        float kk = 0.0f;",
            "        float vv = 0.0f;",
            "        if (key < length) {",
            f"            kk = k[(start + key) * {d}u + e % {d}u];",
            f"            vv = v[(start + key) * {d}u + e % {d}u];",
            "        }",
            "        k_tile[e] = kk;",
            "        v_tile[e] = vv;",
            "    }",
            f"    {BARRIER}",
            f"    float s[{BLOCK_KEYS}];",
            f"    float top = {LOWEST_TEXT};",
            f"    for (uint j = 0u; j < {BLOCK_KEYS}u; ++j) {{",
            "        float dot = 0.0f;",
            f"        for (uint t = 0u; t < {DIMS_PER_PART}u; ++t) {{",
            f"            dot = fma(qv[t], k_tile[j * {d}u + t * {p}u + part], dot);",
            "        }",
            "        dot += simd_shuffle_xor(dot, 1u);",
            "        dot += simd_shuffle_xor(dot, 2u);",
            "        uint key = tile + j;",
            f"        s[j] = dot * {SCALE_TEXT};",
            f"        top = ({valid}) ? max(top, s[j]) : top;",
            "    }",
            "    float next = max(m, top);",
            "    float alpha = precise::exp(m - next);",
            "    l *= alpha;",
            f"    for (uint t = 0u; t < {DIMS_PER_PART}u; ++t) {{",
            "        acc[t] *= alpha;",
            "    }",
            f"    for (uint j = 0u; j < {BLOCK_KEYS}u; ++j) {{",
            "        uint key = tile + j;",
            f"        float w = ({valid}) ? precise::exp(s[j] - next) : 0.0f;",
            "        l += w;",
            f"        for (uint t = 0u; t < {DIMS_PER_PART}u; ++t) {{",
            f"            acc[t] = fma(w, v_tile[j * {d}u + t * {p}u + part], acc[t]);",
            "        }",
            "    }",
            "    m = next;",
            f"    {BARRIER}",
            "}",
            "if (live) {",
            f"    for (uint t = 0u; t < {DIMS_PER_PART}u; ++t) {{",
            f"        o[(start + row) * {d}u + t * {p}u + part] = "
            "precise::divide(acc[t], l);",
            "    }",
            "}",
        ],
    )


_PROBE = _kernel(
    [
        "Probe: y = precise::exp(x), one thread per argument, so a run can",
        "measure the exp error that the bound assumes.",
    ],
    PROBE_EXP,
    ("y", "x"),
    ("uint i [[thread_position_in_grid]]",),
    ["y[i] = precise::exp(x[i]);"],
)


def _registry() -> dict[str, FlashKernel]:
    kernels = []
    for mode in MODES:
        scores, softmax, pv = kernel_names(N0, mode)
        kernels += [
            FlashKernel(
                scores,
                N0,
                mode,
                _SCORES_BUFFERS,
                NAIVE_GROUP,
                HEAD_DIM * FLOAT32_BYTES,
                1,
                "one query row per group; writes its row of the N x N scores",
                _scores(mode),
            ),
            FlashKernel(
                softmax,
                N0,
                mode,
                _SOFTMAX_BUFFERS,
                NAIVE_GROUP,
                2 * _PARTIALS * FLOAT32_BYTES,
                2,
                "reads and writes one row of the N x N buffer twice in place",
                _softmax(mode),
            ),
            FlashKernel(
                pv,
                N0,
                mode,
                _PV_BUFFERS,
                PV_GROUP,
                0,
                0,
                "reads one row of P and the V rows; writes one row of O",
                _pv(mode),
            ),
        ]
    for mode in MODES:
        kernels.append(
            FlashKernel(
                kernel_names(F1, mode)[0],
                F1,
                mode,
                _FLASH_BUFFERS,
                FLASH_GROUP,
                2 * TILE_FLOATS * FLOAT32_BYTES,
                2,
                "one block of 32 queries per group; K/V tiles in threadgroup "
                "memory; online softmax; no N x N buffer",
                _flash(mode),
            )
        )
    kernels.append(
        FlashKernel(
            PROBE_EXP,
            None,
            None,
            ("y", "x"),
            PROBE_GROUP,
            0,
            0,
            "precise::exp at each argument",
            _PROBE,
        )
    )
    return {kernel.name: kernel for kernel in kernels}


KERNELS = _registry()


def flash_library() -> str:
    """Join the header and every kernel into the one text that compiles."""

    return HEADER + "".join("\n" + kernel.source for kernel in KERNELS.values())


def library_sha256() -> str:
    """Hash the exact library text that ``compile_shader`` receives."""

    return text_sha256(flash_library())
