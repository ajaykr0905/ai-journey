"""Pinned CUDA C++ text of the tiled matmul and of a row-major cuBLAS SGEMM.

``tiled_cuda_source`` writes one tiled kernel and its host launcher for a tile
width T of 8, 16 or 32. One T x T block computes one T x T tile of C in
``cdiv(k, T)`` phases. In each phase, each thread loads one element of A and
one element of B into two ``__shared__`` tiles, or 0 for an element outside
the matrix. One ``__syncthreads()`` follows the loads, so no thread reads a
slot before its value is there. A second one follows the inner loop, so no
thread loads the next phase over a value that another thread still reads.
The inner loop runs ``i < min(T, k - phase * T)``, so a partial last phase
reads no padded slot, and ``__fmaf_rn`` rounds each step once, in increasing
k. That is the loop of ``matmul_metal_reference.fma_replay``, so a later run
can check its output bit for bit. Only the store is guarded, because every
thread of a block must reach each barrier.

``cublas_source`` writes one ``cublasSgemm`` call for row-major operands.
cuBLAS reads column-major matrices, and the bytes of a row-major matrix are
the bytes of its transpose in column-major order, so the call computes
``C^T = B^T A^T``. The text sets ``CUBLAS_DEFAULT_MATH`` before the call.
That this mode keeps the call off TF32 Tensor Core math is an assumption from
the cuBLAS documentation, and a later run must check it on the NVIDIA host.

This lab never compiles, links or runs either text. The checkers read a text
back with the Day 73 checker's helpers: comments as a compiler reads them, no
line splice or trigraph, and only the pinned definitions, because one extra
statement, overload or macro can change what a function does. They check the
shared tiles and their size, the loads, the barrier placement, the k order of
the fma, the guards and the host grid. The cuBLAS check also evaluates the
parsed call under the column-major rules, so a swapped argument fails even
when the text is well formed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .kernel_launch_config import LaunchConfig
from .kernel_launch_cuda import (
    _CDIV,
    _DECIMAL,
    _braced,
    _code,
    _expected,
    _function_body,
    _guarded,
    _host_launch,
    _normalize,
    _statements,
    _touches,
)
from .matmul_metal_mapping import MAX_DIMENSION
from .matmul_model_cuda import _FOR, _loop
from .matmul_model_kernels import X_COLUMNS, matmul_launch
from .matmul_model_occupancy import ARCHITECTURE
from .memory_access_tiling import MatmulShape

TILES = (8, 16, 32)
FLOAT_BYTES = 4
CUBLAS_HEADER = "#include <cublas_v2.h>"
TF32_ASSUMPTION = (
    "With CUBLAS_DEFAULT_MATH, cublasSgemm uses no TF32 Tensor Core math. "
    "This is an assumption from the cuBLAS documentation of cublasSetMathMode, "
    "and a later run must check it on the NVIDIA host."
)
TILED_CHECK_NAMES = (
    "cdiv_definition",
    "tiled_signature",
    "tiled_shared_memory",
    "tiled_index",
    "tiled_phase_loop",
    "tiled_loads",
    "tiled_barriers",
    "tiled_inner_loop",
    "tiled_fma_order",
    "tiled_store_guard",
    "tiled_grid",
)
CUBLAS_CHECK_NAMES = (
    "cublas_include",
    "cublas_signature",
    "cublas_scalars",
    "cublas_pointer_mode",
    "cublas_math_mode",
    "cublas_call",
    "cublas_row_major",
)
_SGEMM = (
    "cublasStatus_t sgemm_row_major(cublasHandle_t handle, const float* a, "
    "const float* b, float* c, int m, int n, int k)"
)
_BARRIER = "__syncthreads();"
_STORE_GUARD = "row < m && col < n"
_STORE = "c[row * n + col] = sum;"
_ACCUMULATOR = "float sum = 0.0f;"
_FMA = "sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);"
_LOAD_A = "As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;"
_LOAD_B = "Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;"
_SHARED = re.compile(
    rf"__shared__ float ([A-Za-z_][A-Za-z0-9_]*)\[{_DECIMAL}\]\[{_DECIMAL}\];"
)
_SUM = re.compile(r"(?<![A-Za-z0-9_])sum(?![A-Za-z0-9_])")
_CONTROL = re.compile(
    r"(?<![A-Za-z0-9_])(if|else|return|break|continue|goto|while|do|switch)"
    r"(?![A-Za-z0-9_])"
)
_SCALARS = ("const float alpha = 1.0f;", "const float beta = 0.0f;")
_STATUS_CHECK = "if (status != CUBLAS_STATUS_SUCCESS) { return status; }"
_POINTER_MODE = (
    "cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);",
    _STATUS_CHECK,
)
_MATH_MODE = ("status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);", _STATUS_CHECK)
_SGEMM_CALL = re.compile(
    r"return cublasSgemm\(handle,(CUBLAS_OP_[NT]),(CUBLAS_OP_[NT]),([mnk]),([mnk]),"
    r"([mnk]),&alpha,([abc]),([mnk]),([abc]),([mnk]),&beta,([abc]),([mnk])\);"
)
# Pairwise distinct extents, so a call that swaps two of them reads or writes
# other elements. The second probe permutes the extents of the first.
_PROBES = (MatmulShape(2, 3, 5), MatmulShape(5, 2, 3))


class GpuTiledError(ValueError):
    """Raised when a tiled CUDA lab input, run or report breaks its contract."""


def require_tile(tile: object) -> int:
    """Return a tile width that the pinned text supports: 8, 16 or 32."""

    if type(tile) is not int:
        raise TypeError("tile must be an integer")
    if tile not in TILES:
        raise GpuTiledError(f"tile must be one of {TILES}")
    return tile


def require_shape(shape: object) -> MatmulShape:
    """Return a shape whose int indices stay exact in the kernel.

    With at most 8192 per dimension, every index that the kernel forms, such
    as ``row * k + a_col``, stays below 2**26 and far inside ``int``.
    """

    if type(shape) is not MatmulShape:
        raise TypeError("shape must be MatmulShape")
    if max(shape.m, shape.n, shape.k) > MAX_DIMENSION:
        raise GpuTiledError(f"each dimension must be at most {MAX_DIMENSION}")
    return shape


def shared_memory_bytes(tile: int) -> int:
    """Return the static shared memory of one block: two T x T float tiles."""

    width = require_tile(tile)
    return 2 * width * width * FLOAT_BYTES


def kernel_name(tile: int) -> str:
    return f"tiled_matmul_t{require_tile(tile)}_kernel"


def launcher_name(tile: int) -> str:
    return f"launch_tiled_matmul_t{require_tile(tile)}"


def source_sha256(text: str) -> str:
    """Return the SHA-256 of an ASCII source text."""

    if type(text) is not str or not text.isascii():
        raise GpuTiledError("source text must be an ASCII string")
    return sha256(text.encode("ascii")).hexdigest()


def _kernel_signature(tile: int) -> str:
    return (
        f"__global__ void {kernel_name(tile)}(const float* a, const float* b, "
        "float* c, int m, int n, int k)"
    )


def _launcher_signature(tile: int) -> str:
    return (
        f"void {launcher_name(tile)}(const float* a, const float* b, float* c, "
        "int m, int n, int k)"
    )


def _shared_statements(tile: int) -> tuple[str, str]:
    return (
        f"__shared__ float As[{tile}][{tile}];",
        f"__shared__ float Bs[{tile}][{tile}];",
    )


def _index_statements(tile: int) -> tuple[str, ...]:
    return (
        "int tx = threadIdx.x;",
        "int ty = threadIdx.y;",
        f"int row = blockIdx.y * {tile} + ty;",
        f"int col = blockIdx.x * {tile} + tx;",
    )


def _phase_loop(tile: int) -> str:
    return f"for (int phase = 0; phase * {tile} < k; ++phase)"


def _load_statements(tile: int) -> tuple[str, ...]:
    return (
        f"int a_col = phase * {tile} + tx;",
        f"int b_row = phase * {tile} + ty;",
        _LOAD_A,
        _LOAD_B,
    )


def _inner_loop(tile: int) -> str:
    return f"for (int i = 0; i < min({tile}, k - phase * {tile}); ++i)"


def tiled_cuda_source(*, tile: int) -> str:
    """Return the pinned CUDA C++ text of the tiled kernel for one tile width."""

    width = require_tile(tile)
    shared_a, shared_b = _shared_statements(width)
    tx, ty, row, col = _index_statements(width)
    a_col, b_row, load_a, load_b = _load_statements(width)
    return f"""\
// CUDA C++ source generated by ai_journey.gpu_tiled_cuda.
// The lab that generated this text did not compile or run it.
// It is source for a later run on an NVIDIA GPU.
// One {width} x {width} block computes one {width} x {width} tile of C in cdiv(k, {width}) phases.
// In each phase, each thread loads one element of A and one of B into
// shared memory, or 0 outside the matrix. A barrier follows the loads, and a
// second barrier follows the inner loop. The inner loop stops at
// min({width}, k - phase * {width}), and __fmaf_rn rounds each step once, in
// increasing k. Only the store is guarded, so every thread reaches both
// barriers.

{_CDIV} {{
    return (a + b - 1) / b;
}}

__global__ void {kernel_name(width)}(
    const float* a, const float* b, float* c, int m, int n, int k) {{
    {shared_a}
    {shared_b}
    {tx}
    {ty}
    {row}
    {col}
    {_ACCUMULATOR}
    {_phase_loop(width)} {{
        {a_col}
        {b_row}
        {load_a}
        {load_b}
        {_BARRIER}
        {_inner_loop(width)} {{
            {_FMA}
        }}
        {_BARRIER}
    }}
    if ({_STORE_GUARD}) {{
        {_STORE}
    }}
}}

void {launcher_name(width)}(
    const float* a, const float* b, float* c, int m, int n, int k) {{
    dim3 block({width}, {width});
    dim3 grid(cdiv(n, {width}), cdiv(m, {width}));
    {kernel_name(width)}<<<grid, block>>>(a, b, c, m, n, k);
}}
"""


def cublas_source() -> str:
    """Return the pinned text of one row-major ``cublasSgemm`` call."""

    return f"""\
// cuBLAS call text generated by ai_journey.gpu_tiled_cuda.
// The lab that generated this text did not compile or link it, and it did
// not call cuBLAS. It is source for a later run on an NVIDIA GPU.
// cuBLAS reads column-major matrices. The bytes of the row-major m x n
// matrix C are the bytes of the column-major n x m matrix C^T. Thus the call
// computes C^T = B^T A^T: B^T is n x k with leading dimension n, A^T is
// k x m with leading dimension k, and C^T has leading dimension n.
// ASSUMPTION from the cuBLAS documentation, to check on the NVIDIA host:
// with CUBLAS_DEFAULT_MATH, cublasSgemm uses no TF32 Tensor Core math.

{CUBLAS_HEADER}

cublasStatus_t sgemm_row_major(
    cublasHandle_t handle, const float* a, const float* b, float* c, int m,
    int n, int k) {{
    {_SCALARS[0]}
    {_SCALARS[1]}
    {_POINTER_MODE[0]}
    if (status != CUBLAS_STATUS_SUCCESS) {{
        return status;
    }}
    {_MATH_MODE[0]}
    if (status != CUBLAS_STATUS_SUCCESS) {{
        return status;
    }}
    return cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta,
        c, n);
}}
"""


def _definitions(source: str, signatures: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the function bodies if the text is exactly these definitions."""

    bodies = [_function_body(source, signature) for signature in signatures]
    if any(body is None for body in bodies):
        return None
    rebuilt = "".join(
        f"{_normalize(signature)}{{{body}}}"
        for signature, body in zip(signatures, bodies)
    )
    return tuple(bodies) if rebuilt == source else None


def _first_loop(code: str) -> tuple[str, str, str, str] | None:
    """Split code at its first ``for`` loop: before, header, body and after.

    ``matmul_model_cuda._loop`` needs the one loop of its code; the phase
    loop holds the inner loop, so this split takes the first one.
    """

    found = _FOR.search(code)
    if found is None:
        return None
    depth = 0
    for position in range(found.end() - 1, len(code)):
        if code[position] == "(":
            depth += 1
        elif code[position] == ")":
            depth -= 1
            if depth == 0:
                break
    else:
        return None
    opening = position + 1
    body = _braced(code, opening) if code[opening : opening + 1] == "{" else None
    if body is None:
        return None
    start = found.start()
    return code[:start], code[start:opening], body, code[opening + len(body) + 2 :]


def _shared_bytes(kernel: str) -> tuple[tuple[str, int, int], ...] | None:
    """Return each ``__shared__ float`` tile and its extents, or ``None``.

    Any other ``__shared__`` declaration, such as an ``extern`` array, makes
    the size unknown, so it returns ``None``.
    """

    tiles = tuple(
        (name, int(rows), int(cols)) for name, rows, cols in _SHARED.findall(kernel)
    )
    return tiles if len(tiles) == kernel.count("__shared__") else None


@dataclass(frozen=True)
class TiledCudaCheck:
    """What a compiler-free reading of one tiled kernel text confirmed."""

    tile: int
    checks: tuple[tuple[str, bool], ...]
    shared_memory_bytes: int | None
    launch: LaunchConfig | None

    @property
    def passed(self) -> bool:
        return all(result for _, result in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.checks if not result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "tile": self.tile,
            "checks": dict(self.checks),
            "passed": self.passed,
            "shared_memory_bytes": self.shared_memory_bytes,
            "launch": None if self.launch is None else self.launch.to_mapping(),
        }


def _launcher_pattern(tile: int) -> re.Pattern[str]:
    return re.compile(
        rf"dim3 block\({_DECIMAL},{_DECIMAL}\);"
        rf"dim3 grid\(cdiv\(n,{_DECIMAL}\),cdiv\(m,{_DECIMAL}\)\);"
        rf"{kernel_name(tile)}<<<grid,block>>>\(a,b,c,m,n,k\);"
    )


def _phase_checks(tile: int, before: str) -> dict[str, bool]:
    """Check the code before the store guard: the tiles, indices and phases.

    Statements are sorted by kind before each comparison, so an extra or a
    changed statement fails the check of its own part: a ``__shared__``
    declaration the tile check, a statement that names ``sum`` the fma check,
    and a barrier the barrier check.
    """

    checks: dict[str, bool] = {}
    split = _first_loop(before)
    if split is None:
        return checks
    prelude, header, body, rest = split
    declared = _statements(prelude)
    shared = tuple(item for item in declared if item.startswith("__shared__"))
    sums = tuple(item for item in declared if _SUM.search(item))
    index = tuple(item for item in declared if item not in shared + sums)
    checks["tiled_shared_memory"] = shared == _expected(*_shared_statements(tile))
    checks["tiled_index"] = index == _expected(*_index_statements(tile))
    checks["tiled_phase_loop"] = header == _normalize(_phase_loop(tile)) and not rest
    inner = _loop(body)
    if inner is None:
        return checks
    loads, inner_header, inner_body, after = inner
    staged = _statements(loads)
    barrier = _expected(_BARRIER)
    checks["tiled_loads"] = tuple(
        item for item in staged if item not in barrier
    ) == _expected(*_load_statements(tile))
    checks["tiled_barriers"] = (
        staged[-1:] == barrier
        and staged.count(barrier[0]) == 1
        and _statements(after) == barrier
    )
    checks["tiled_inner_loop"] = inner_header == _normalize(_inner_loop(tile))
    checks["tiled_fma_order"] = sums == _expected(_ACCUMULATOR) and _statements(
        inner_body
    ) == _expected(_FMA)
    return checks


def check_tiled_source(text: str, *, tile: int, shape: MatmulShape) -> TiledCudaCheck:
    """Check one tiled kernel text and its host grid for one shape.

    The kernel must be its shared tiles, index statements and accumulator,
    then the phase loop, then one guard that holds only the store. The phase
    loop must hold the two loads, a barrier, the inner loop with its one fma,
    and a second barrier, and the kernel no other barrier. The size check
    reads the extents of each ``__shared__`` tile from the text and requires
    ``2 * T * T * 4`` bytes. The grid check evaluates the host ``dim3``
    expressions with C unsigned division and requires the launch that the
    Day 74 model gives a ``(T, T)`` block.
    """

    if type(text) is not str:
        raise TypeError("CUDA source must be a string")
    width = require_tile(tile)
    checked = require_shape(shape)
    expected_launch = matmul_launch(checked, (width, width), X_COLUMNS)
    code = _code(text)
    signatures = (_CDIV, _kernel_signature(width), _launcher_signature(width))
    bodies = None if code is None else _definitions(_normalize(code), signatures)
    if bodies is None:
        return TiledCudaCheck(
            tile=width,
            checks=tuple((name, False) for name in TILED_CHECK_NAMES),
            shared_memory_bytes=None,
            launch=None,
        )
    cdiv, kernel, host = bodies
    tiles = _shared_bytes(kernel)
    shared_bytes = (
        None
        if tiles is None
        else sum(rows * cols * FLOAT_BYTES for _, rows, cols in tiles)
    )
    checks = dict.fromkeys(TILED_CHECK_NAMES, False)
    checks["cdiv_definition"] = cdiv == _normalize("return (a + b - 1) / b;")
    # _definitions found each signature exactly once.
    checks["tiled_signature"] = True
    guarded = _guarded(kernel, _STORE_GUARD)
    if guarded is not None:
        before, inside, after = guarded
        checks.update(_phase_checks(width, before))
        checks["tiled_store_guard"] = (
            _statements(inside) == _expected(_STORE)
            and not after
            and _CONTROL.findall(kernel) == ["if"]
            and not _touches(before, "c")
        )
    # A barrier inside the guard would be a barrier that some threads skip.
    checks["tiled_barriers"] = (
        checks["tiled_barriers"] and kernel.count("__syncthreads") == 2
    )
    checks["tiled_shared_memory"] = (
        checks["tiled_shared_memory"]
        and tiles == (("As", width, width), ("Bs", width, width))
        and shared_bytes == shared_memory_bytes(width)
        and shared_bytes <= ARCHITECTURE.shared_memory_per_block_bytes
    )
    launch = _host_launch(host, _launcher_pattern(width), (checked.n, checked.m))
    checks["tiled_grid"] = launch == expected_launch
    return TiledCudaCheck(
        tile=width,
        checks=tuple((name, bool(checks[name])) for name in TILED_CHECK_NAMES),
        shared_memory_bytes=shared_bytes,
        launch=launch,
    )


def require_tiled_source(text: str, *, tile: int, shape: MatmulShape) -> TiledCudaCheck:
    """Return the check of ``text``, or raise naming every failed check."""

    result = check_tiled_source(text, tile=tile, shape=shape)
    if not result.passed:
        raise GpuTiledError(
            "tiled CUDA source fails structural checks: " + ", ".join(result.failures)
        )
    return result


@dataclass(frozen=True)
class SgemmCall:
    """The arguments of one parsed ``cublasSgemm`` call, as the text names them."""

    transa: str
    transb: str
    m: str
    n: str
    k: str
    a: str
    lda: str
    b: str
    ldb: str
    c: str
    ldc: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "transa": self.transa,
            "transb": self.transb,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "a": self.a,
            "lda": self.lda,
            "b": self.b,
            "ldb": self.ldb,
            "c": self.c,
            "ldc": self.ldc,
        }


def _operand_extents(call: SgemmCall, extents: dict[str, int]) -> dict[str, int]:
    """Return the stored rows and columns of each column-major operand."""

    rows, cols, inner = extents[call.m], extents[call.n], extents[call.k]
    a_rows, a_cols = (rows, inner) if call.transa == "CUBLAS_OP_N" else (inner, rows)
    b_rows, b_cols = (inner, cols) if call.transb == "CUBLAS_OP_N" else (cols, inner)
    return {
        "rows": rows,
        "cols": cols,
        "inner": inner,
        "a_rows": a_rows,
        "a_cols": a_cols,
        "b_rows": b_rows,
        "b_cols": b_cols,
    }


def call_fits(call: SgemmCall, shape: MatmulShape) -> bool:
    """Return whether the call's leading dimensions and extents fit the buffers.

    cuBLAS refuses a leading dimension below the stored row count of its
    operand. Each buffer also holds only its row-major matrix: ``a`` has
    ``m * k`` elements, ``b`` has ``k * n`` and ``c`` has ``m * n``, so the
    last element that a column-major view reads must lie inside it.
    """

    extents = {"m": shape.m, "n": shape.n, "k": shape.k}
    sizes = {"a": shape.m * shape.k, "b": shape.k * shape.n, "c": shape.m * shape.n}
    view = _operand_extents(call, extents)
    operands = (
        (call.a, view["a_rows"], view["a_cols"], extents[call.lda]),
        (call.b, view["b_rows"], view["b_cols"], extents[call.ldb]),
        (call.c, view["rows"], view["cols"], extents[call.ldc]),
    )
    return all(
        lead >= max(1, rows) and (rows - 1) + (cols - 1) * lead < sizes[buffer]
        for buffer, rows, cols, lead in operands
    )


def _element(buffer: list[int], transposed: bool, row: int, col: int, lead: int) -> int:
    """Return ``op(X)[row, col]`` of a column-major ``X`` with leading ``lead``."""

    return buffer[col + row * lead] if transposed else buffer[row + col * lead]


def column_major_sgemm(
    call: SgemmCall, shape: MatmulShape, a: list[int], b: list[int]
) -> list[int] | None:
    """Evaluate the parsed call as cuBLAS defines it, with exact integers.

    ``a`` and ``b`` hold row-major A and B. Each operand pointer of the call
    reads its buffer as a column-major matrix with its leading dimension,
    and ``alpha = 1``, ``beta = 0``. The result is the buffer ``c`` after the
    call, or ``None`` when the call does not fit its buffers or writes into
    an input.
    """

    if not call_fits(call, shape) or call.c != "c" or "c" in (call.a, call.b):
        return None
    extents = {"m": shape.m, "n": shape.n, "k": shape.k}
    view = _operand_extents(call, extents)
    buffers = {"a": a, "b": b}
    left, right = buffers[call.a], buffers[call.b]
    lda, ldb, ldc = extents[call.lda], extents[call.ldb], extents[call.ldc]
    left_t, right_t = call.transa == "CUBLAS_OP_T", call.transb == "CUBLAS_OP_T"
    output = [0] * (shape.m * shape.n)
    for j in range(view["cols"]):
        for i in range(view["rows"]):
            output[i + j * ldc] = sum(
                _element(left, left_t, i, p, lda) * _element(right, right_t, p, j, ldb)
                for p in range(view["inner"])
            )
    return output


def _probe_operands(shape: MatmulShape) -> tuple[list[int], list[int]]:
    a = [(7 * index + 3) % 11 - 5 for index in range(shape.m * shape.k)]
    b = [(5 * index + 2) % 13 - 6 for index in range(shape.k * shape.n)]
    return a, b


def computes_row_major(call: SgemmCall) -> bool:
    """Return whether the call leaves row-major ``C = A @ B`` in ``c``.

    The call pattern names 78,732 calls: two operations for each operand,
    ``m``, ``n`` or ``k`` for each extent and leading dimension, and ``a``,
    ``b`` or ``c`` for each pointer. Only the ``C^T = B^T A^T`` call gives
    the exact product on both probes, and a test enumerates all of them.
    """

    for probe in _PROBES:
        a, b = _probe_operands(probe)
        expected = [
            sum(a[row * probe.k + p] * b[p * probe.n + col] for p in range(probe.k))
            for row in range(probe.m)
            for col in range(probe.n)
        ]
        if column_major_sgemm(call, probe, a, b) != expected:
            return False
    return True


def _top_statements(code: str) -> tuple[str, ...] | None:
    """Split normalized code into statements; an ``if`` block stays one."""

    statements: list[str] = []
    start = braces = parentheses = 0
    for position, character in enumerate(code):
        if character == "(":
            parentheses += 1
        elif character == ")":
            parentheses -= 1
        elif character == "{":
            braces += 1
        elif character == "}":
            braces -= 1
            if braces == 0 and parentheses == 0:
                statements.append(code[start : position + 1])
                start = position + 1
        elif character == ";" and braces == 0 and parentheses == 0:
            statements.append(code[start : position + 1])
            start = position + 1
        if braces < 0 or parentheses < 0:
            return None
    return tuple(statements) if start == len(code) else None


def _cublas_body(code: str) -> tuple[bool, str] | None:
    """Return whether the text opens with the one include, and the code after.

    A directive other than that include, such as a macro that renames
    ``CUBLAS_DEFAULT_MATH``, can change the call, so it fails the whole text.
    """

    lines = code.splitlines()
    directives = [line.strip() for line in lines if line.lstrip().startswith("#")]
    if directives not in ([], [CUBLAS_HEADER]):
        return None
    leading = next((line.strip() for line in lines if line.strip()), "")
    body = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return leading == CUBLAS_HEADER, body


@dataclass(frozen=True)
class CublasCheck:
    """What a compiler-free reading of the cuBLAS call text confirmed."""

    checks: tuple[tuple[str, bool], ...]
    call: SgemmCall | None

    @property
    def passed(self) -> bool:
        return all(result for _, result in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.checks if not result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "checks": dict(self.checks),
            "passed": self.passed,
            "call": None if self.call is None else self.call.to_mapping(),
        }


def check_cublas_source(text: str, *, shape: MatmulShape) -> CublasCheck:
    """Check the cuBLAS call text, and its call for one shape.

    The function must set the host pointer mode and ``CUBLAS_DEFAULT_MATH``,
    return each failed status, and then return one ``cublasSgemm`` call. The
    row-major check evaluates the parsed call under the column-major rules on
    two probes and requires ``C = A @ B``, and it requires that the leading
    dimensions of the call fit the buffers of ``shape``.
    """

    if type(text) is not str:
        raise TypeError("cuBLAS source must be a string")
    checked = require_shape(shape)
    code = _code(text)
    split = None if code is None else _cublas_body(code)
    bodies = None if split is None else _definitions(_normalize(split[1]), (_SGEMM,))
    statements = None if bodies is None else _top_statements(bodies[0])
    if statements is None:
        return CublasCheck(
            checks=tuple((name, False) for name in CUBLAS_CHECK_NAMES), call=None
        )
    found = _SGEMM_CALL.fullmatch(statements[-1]) if len(statements) == 7 else None
    call = None if found is None else SgemmCall(*found.groups())
    math_modes = re.findall(r"cublasSetMathMode|TF32", bodies[0])
    checks = {
        "cublas_include": split[0],
        # _definitions found the signature exactly once.
        "cublas_signature": True,
        "cublas_scalars": statements[:2] == _expected(*_SCALARS),
        "cublas_pointer_mode": statements[2:4] == _expected(*_POINTER_MODE),
        "cublas_math_mode": statements[4:6] == _expected(*_MATH_MODE)
        and math_modes == ["cublasSetMathMode"],
        "cublas_call": call is not None and bodies[0].count("cublasSgemm") == 1,
        "cublas_row_major": call is not None
        and computes_row_major(call)
        and call_fits(call, checked),
    }
    return CublasCheck(
        checks=tuple((name, bool(checks[name])) for name in CUBLAS_CHECK_NAMES),
        call=call,
    )


def require_cublas_source(text: str, *, shape: MatmulShape) -> CublasCheck:
    """Return the check of ``text``, or raise naming every failed check."""

    result = check_cublas_source(text, shape=shape)
    if not result.passed:
        raise GpuTiledError(
            "cuBLAS source fails structural checks: " + ", ".join(result.failures)
        )
    return result
