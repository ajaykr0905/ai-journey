"""Metal Shading Language source of the Day 77 checklist kernels, from one template.

Each kernel is a block matmul: one threadgroup computes one ``BM x BN`` tile
of ``C = A @ B``, all three row-major float32, and walks K in phases of
``BK``. Each thread keeps a ``TM x TN`` block of outputs in registers, at
rows ``row0 + local.y + GY * i`` and columns ``column0 + local.x + GX * j``,
so the threads of one SIMD group store neighbouring columns together. A
``BlockDesign`` fixes every size and three switches, and the generator turns
it into text, so a checklist step changes one field and nothing else.

These rules keep each stored output equal to ``fma_replay`` in principle:

- Each output takes exactly K steps of ``sum = fma(a, b, sum)`` in increasing
  k from 0. A guarded phase runs ``min(BK, k - start)`` steps, so a
  zero-filled slot past K is never read: ``fma(0, 0, -0.0)`` is ``+0.0``.
- A slot past M (in A) or past N (in B) holds 0 and feeds only outputs that
  are never stored.
- Only stores are guarded. Every barrier sits in the phase body, outside any
  condition on the thread, so every thread of a group reaches it. The only
  conditions around a barrier are uniform over the group: the interior test
  of the fast path and the K tail.

The switches:

- ``vector_loads``: stage four consecutive floats with one ``float4`` load and
  one ``float4`` store when K (for A) or N (for B) is a multiple of 4, and on
  the scalar edge path otherwise. BK and BN must be multiples of 4 too, so
  every vector address is 16-byte aligned and each float4 holds four floats
  of one tile row. A float4 that lies past the matrix is zero.
- ``pad``: add ``pad`` floats to each tile row in threadgroup memory, a bank
  conflict control. Each padded row holds whole float4 values, so rows stay
  16-byte aligned, and with vector loads the pad is a multiple of 4.
- ``interior_path``: a group whose tile lies inside M and N, with K and N
  multiples of 4, runs each full phase with no zero-fill select and a
  constant loop bound, and stores with no guard. Edge groups and the K tail
  run the guarded code. The test is uniform over the group.

Every kernel declares its tiles as ``float4`` arrays, viewed as ``float`` for
the scalar reads, so a vector store is always aligned and a design without
vector loads has the same memory as one with them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .kernel_metal_geometry import MAX_THREADGROUP_THREADS, require_integer
from .matmul_metal_mapping import FLOAT32_BYTES
from .matmul_tiled_source import BARRIER, MatmulTiledError

# Metal guarantees at least this much static threadgroup memory per group.
MAX_THREADGROUP_BYTES = 32 * 1024
MAX_BLOCK = 256
MAX_BLOCK_K = 64
MAX_THREAD_OUTPUTS = 8
MAX_PAD = 16
VECTOR_WIDTH = 4
LOOP_STATEMENT = "sum_{i}_{j} = fma(a_{i}, b_{j}, sum_{i}_{j});"
STORE_GUARD = "if (row_{i} < m && column_{j} < n) {{"
INTERIOR_TEST = (
    "bool interior = vector_a && vector_b && row0 + {bm}u <= m && "
    "column0 + {bn}u <= n;"
)
_NAME = re.compile(r"matmul_[a-z0-9_]{1,48}")


class MatmulChecklistError(MatmulTiledError):
    """Raised when a checklist kernel, control, launch, check or report is invalid."""


@dataclass(frozen=True)
class BlockDesign:
    """Every size and switch of one block kernel, checked before any text exists."""

    name: str
    block_m: int
    block_n: int
    block_k: int
    thread_m: int
    thread_n: int
    vector_loads: bool
    pad: int
    interior_path: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MatmulChecklistError("a kernel name must be matmul_ and a short slug")
        require_integer(self.block_m, "block_m", 1, MAX_BLOCK)
        require_integer(self.block_n, "block_n", 1, MAX_BLOCK)
        require_integer(self.block_k, "block_k", 1, MAX_BLOCK_K)
        require_integer(self.thread_m, "thread_m", 1, MAX_THREAD_OUTPUTS)
        require_integer(self.thread_n, "thread_n", 1, MAX_THREAD_OUTPUTS)
        require_integer(self.pad, "pad", 0, MAX_PAD)
        for flag in ("vector_loads", "interior_path"):
            if type(getattr(self, flag)) is not bool:
                raise TypeError(f"{flag} must be a bool")
        if self.block_m % self.thread_m or self.block_n % self.thread_n:
            raise MatmulChecklistError("each tile side must be whole thread blocks")
        if self.threads_per_group > MAX_THREADGROUP_THREADS:
            raise MatmulChecklistError(
                f"a group of {self.threads_per_group} threads exceeds "
                f"{MAX_THREADGROUP_THREADS}"
            )
        # float4 storage needs every row to hold whole float4 values.
        if self.a_stride % VECTOR_WIDTH or self.b_stride % VECTOR_WIDTH:
            raise MatmulChecklistError("each tile row must hold whole float4 values")
        # A float4 stage moves four floats of one row of data, not of padding:
        # with BK or BN off a multiple of 4 it would skip columns and load from
        # addresses that are not 16-byte aligned.
        if self.vector_loads and (
            self.block_k % VECTOR_WIDTH or self.block_n % VECTOR_WIDTH
        ):
            raise MatmulChecklistError(
                "float4 staging needs BK and BN to be multiples of 4"
            )
        if self.interior_path and not self.vector_loads:
            raise MatmulChecklistError("the interior path needs vector loads")
        if self.threadgroup_bytes > MAX_THREADGROUP_BYTES:
            raise MatmulChecklistError(
                f"{self.name} needs {self.threadgroup_bytes} bytes of threadgroup "
                f"memory; the bound is {MAX_THREADGROUP_BYTES}"
            )

    @property
    def group_size(self) -> tuple[int, int]:
        """The ``(x, y)`` threadgroup: ``(BN / TN, BM / TM)``."""

        return (self.block_n // self.thread_n, self.block_m // self.thread_m)

    @property
    def threads_per_group(self) -> int:
        """Threads in one group: ``(BM / TM) * (BN / TN)``."""

        x, y = self.group_size
        return x * y

    @property
    def a_stride(self) -> int:
        """Floats in one row of the A tile in threadgroup memory: ``BK + pad``."""

        return self.block_k + self.pad

    @property
    def b_stride(self) -> int:
        """Floats in one row of the B tile in threadgroup memory: ``BN + pad``."""

        return self.block_n + self.pad

    @property
    def threadgroup_bytes(self) -> int:
        """Static threadgroup memory of both tiles, padding included."""

        floats = self.block_m * self.a_stride + self.block_k * self.b_stride
        return floats * FLOAT32_BYTES

    def to_mapping(self) -> dict[str, object]:
        """Return the design as a report records it."""

        return {
            "block": [self.block_m, self.block_n, self.block_k],
            "outputs_per_thread": [self.thread_m, self.thread_n],
            "vector_loads": self.vector_loads,
            "pad": self.pad,
            "interior_path": self.interior_path,
        }


def _signature(name: str) -> list[str]:
    signature = f"kernel void {name}("
    indent = " " * len(signature)
    parameters = f",\n{indent}".join(
        (
            "device float* c [[buffer(0)]]",
            "device const float* a [[buffer(1)]]",
            "device const float* b [[buffer(2)]]",
            "constant uint& m [[buffer(3)]]",
            "constant uint& n [[buffer(4)]]",
            "constant uint& k [[buffer(5)]]",
            "uint2 group [[threadgroup_position_in_grid]]",
            "uint2 local [[thread_position_in_threadgroup]]",
            "uint index [[thread_index_in_threadgroup]]",
        )
    )
    return f"{signature}{parameters}) {{".splitlines()


def _chunks(count: int, threads: int) -> list[tuple[int, int | None]]:
    """Split ``count`` staged values over the group: ``(offset, limit)`` pairs.

    A full chunk gives one value to every thread. A last partial chunk gives
    one value to each thread below ``limit``; that condition only shares out
    the staging, and no barrier lies inside it.
    """

    chunks: list[tuple[int, int | None]] = [
        (offset, None) for offset in range(0, count - count % threads, threads)
    ]
    if count % threads:
        chunks.append((count - count % threads, count % threads))
    return chunks


def _chunk(lines: list[str], limit: int | None, indent: str) -> list[str]:
    head = f"if (index < {limit}u) {{" if limit is not None else "{"
    return [indent + head, *(f"{indent}    {line}" for line in lines), indent + "}"]


def _scalar_a(design: BlockDesign, indent: str) -> list[str]:
    lines = []
    for offset, limit in _chunks(
        design.block_m * design.block_k, design.threads_per_group
    ):
        body = [
            f"uint e = index + {offset}u;",
            f"uint r = e / {design.block_k}u;",
            f"uint q = e % {design.block_k}u;",
            "uint a_row = row0 + r;",
            "uint a_column = start + q;",
            f"a_tile[r * {design.a_stride}u + q] =",
            "    a_row < m && a_column < k ? a[a_row * k + a_column] : 0.0f;",
        ]
        lines += _chunk(body, limit, indent)
    return lines


def _scalar_b(design: BlockDesign, indent: str) -> list[str]:
    lines = []
    for offset, limit in _chunks(
        design.block_k * design.block_n, design.threads_per_group
    ):
        body = [
            f"uint e = index + {offset}u;",
            f"uint q = e / {design.block_n}u;",
            f"uint s = e % {design.block_n}u;",
            "uint b_row = start + q;",
            "uint b_column = column0 + s;",
            f"b_tile[q * {design.b_stride}u + s] =",
            "    b_row < k && b_column < n ? b[b_row * n + b_column] : 0.0f;",
        ]
        lines += _chunk(body, limit, indent)
    return lines


def _vector_a(design: BlockDesign, indent: str, guarded: bool) -> list[str]:
    per_row = design.block_k // VECTOR_WIDTH
    lines = []
    for offset, limit in _chunks(design.block_m * per_row, design.threads_per_group):
        load = "*(device const float4*)(a + a_row * k + a_column)"
        target = f"a_store[r * {design.a_stride // VECTOR_WIDTH}u + q] ="
        store = (
            [
                target,
                "    a_row < m && a_column < k",
                f"    ? {load}",
                "    : float4(0.0f);",
            ]
            if guarded
            else [f"{target} {load};"]
        )
        body = [
            f"uint e = index + {offset}u;",
            f"uint r = e / {per_row}u;",
            f"uint q = e % {per_row}u;",
            "uint a_row = row0 + r;",
            "uint a_column = start + 4u * q;",
            *store,
        ]
        lines += _chunk(body, limit, indent)
    return lines


def _vector_b(design: BlockDesign, indent: str, guarded: bool) -> list[str]:
    per_row = design.block_n // VECTOR_WIDTH
    lines = []
    for offset, limit in _chunks(design.block_k * per_row, design.threads_per_group):
        load = "*(device const float4*)(b + b_row * n + b_column)"
        target = f"b_store[q * {design.b_stride // VECTOR_WIDTH}u + s] ="
        store = (
            [
                target,
                "    b_row < k && b_column < n",
                f"    ? {load}",
                "    : float4(0.0f);",
            ]
            if guarded
            else [f"{target} {load};"]
        )
        body = [
            f"uint e = index + {offset}u;",
            f"uint q = e / {per_row}u;",
            f"uint s = e % {per_row}u;",
            "uint b_row = start + q;",
            "uint b_column = column0 + 4u * s;",
            *store,
        ]
        lines += _chunk(body, limit, indent)
    return lines


def _stage(design: BlockDesign, indent: str, guarded: bool = True) -> list[str]:
    """Stage one phase of both tiles; an unguarded stage is the interior one."""

    if not design.vector_loads:
        return _scalar_a(design, indent) + _scalar_b(design, indent)
    if not guarded:
        return _vector_a(design, indent, False) + _vector_b(design, indent, False)
    inner = indent + "    "
    return [
        f"{indent}if (vector_a) {{",
        *_vector_a(design, inner, True),
        f"{indent}}} else {{",
        *_scalar_a(design, inner),
        f"{indent}}}",
        f"{indent}if (vector_b) {{",
        *_vector_b(design, inner, True),
        f"{indent}}} else {{",
        *_scalar_b(design, inner),
        f"{indent}}}",
    ]


def _compute(design: BlockDesign, indent: str, bound: str) -> list[str]:
    """The k loop of one phase: TM + TN reads, then one fma per output."""

    x, y = design.group_size
    inner = indent + "    "
    reads = [
        f"{inner}float a_{i} = a_tile[a_base + {y * i * design.a_stride}u + p];"
        for i in range(design.thread_m)
    ]
    reads += [
        f"{inner}float b_{j} = b_tile[p * {design.b_stride}u + local.x + {x * j}u];"
        for j in range(design.thread_n)
    ]
    steps = [
        inner + LOOP_STATEMENT.format(i=i, j=j)
        for i in range(design.thread_m)
        for j in range(design.thread_n)
    ]
    return [
        f"{indent}for (uint p = 0; p < {bound}; ++p) {{",
        *reads,
        *steps,
        f"{indent}}}",
    ]


def _phase(design: BlockDesign, indent: str, bound: str, guarded: bool) -> list[str]:
    head = (
        [f"{indent}uint steps = min({design.block_k}u, k - start);"]
        if bound == "steps"
        else []
    )
    return [
        *_stage(design, indent, guarded),
        f"{indent}{BARRIER}",
        *head,
        *_compute(design, indent, bound),
        f"{indent}{BARRIER}",
    ]


def _store(design: BlockDesign, indent: str, guarded: bool) -> list[str]:
    lines = []
    for i in range(design.thread_m):
        for j in range(design.thread_n):
            store = f"c[row_{i} * n + column_{j}] = sum_{i}_{j};"
            if guarded:
                lines += [
                    indent + STORE_GUARD.format(i=i, j=j),
                    f"{indent}    {store}",
                    indent + "}",
                ]
            else:
                lines.append(indent + store)
    return lines


def _guarded_body(design: BlockDesign, indent: str) -> list[str]:
    inner = indent + "    "
    return [
        f"{indent}for (uint start = 0; start < k; start += {design.block_k}u) {{",
        *_phase(design, inner, "steps", True),
        f"{indent}}}",
        *_store(design, indent, True),
    ]


def _interior_body(design: BlockDesign, indent: str) -> list[str]:
    inner = indent + "    "
    block_k = design.block_k
    return [
        f"{indent}uint start = 0;",
        f"{indent}for (; start + {block_k}u <= k; start += {block_k}u) {{",
        *_phase(design, inner, f"{block_k}u", False),
        f"{indent}}}",
        f"{indent}if (start < k) {{",
        *_phase(design, inner, "steps", True),
        f"{indent}}}",
        *_store(design, indent, False),
    ]


def block_kernel_source(design: BlockDesign) -> str:
    """Return the kernel text of one design, without the library header."""

    if not isinstance(design, BlockDesign):
        raise TypeError("design must be a BlockDesign")
    x, y = design.group_size
    a_vectors = design.block_m * design.a_stride // VECTOR_WIDTH
    b_vectors = design.block_k * design.b_stride // VECTOR_WIDTH
    switches = []
    if design.vector_loads:
        switches = [
            "    bool vector_a = (k & 3u) == 0u;",
            "    bool vector_b = (n & 3u) == 0u;",
        ]
    head = [
        f"// One {x} x {y} group per {design.block_m} x {design.block_n} tile of "
        f"C = A @ B, in K phases of {design.block_k}.",
        f"// Thread (x, y) holds the outputs at rows row0 + y + {y} i and columns",
        f"// column0 + x + {x} j, {design.thread_m} x {design.thread_n} in "
        "registers; only the stores are guarded.",
        *_signature(design.name),
        f"    threadgroup float4 a_store[{a_vectors}];",
        f"    threadgroup float4 b_store[{b_vectors}];",
        "    threadgroup float* a_tile = (threadgroup float*)a_store;",
        "    threadgroup float* b_tile = (threadgroup float*)b_store;",
        f"    uint row0 = group.y * {design.block_m}u;",
        f"    uint column0 = group.x * {design.block_n}u;",
        f"    uint a_base = local.y * {design.a_stride}u;",
        *switches,
        *(
            f"    float sum_{i}_{j} = 0.0f;"
            for i in range(design.thread_m)
            for j in range(design.thread_n)
        ),
        *(
            f"    uint row_{i} = row0 + local.y + {y * i}u;"
            for i in range(design.thread_m)
        ),
        *(
            f"    uint column_{j} = column0 + local.x + {x * j}u;"
            for j in range(design.thread_n)
        ),
    ]
    if design.interior_path:
        body = [
            "    " + INTERIOR_TEST.format(bm=design.block_m, bn=design.block_n),
            "    if (interior) {",
            *_interior_body(design, "        "),
            "    } else {",
            *_guarded_body(design, "        "),
            "    }",
        ]
    else:
        body = _guarded_body(design, "    ")
    return "\n".join([*head, *body, "}"]) + "\n"
