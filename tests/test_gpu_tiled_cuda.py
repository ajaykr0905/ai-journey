import itertools
import re
from hashlib import sha256

import pytest

from ai_journey.gpu_tiled_cuda import (
    CUBLAS_CHECK_NAMES,
    CUBLAS_HEADER,
    TF32_ASSUMPTION,
    TILED_CHECK_NAMES,
    TILES,
    GpuTiledError,
    SgemmCall,
    call_fits,
    check_cublas_source,
    check_tiled_source,
    column_major_sgemm,
    computes_row_major,
    cublas_source,
    kernel_name,
    launcher_name,
    require_cublas_source,
    require_tiled_source,
    shared_memory_bytes,
    source_sha256,
    tiled_cuda_source,
)
from ai_journey.kernel_launch_config import Dim3, LaunchConfig
from ai_journey.matmul_model_kernels import X_COLUMNS, matmul_launch
from ai_journey.memory_access_tiling import MatmulShape

# The correctness shapes of config/day-74-matmul-metal.json, in its order.
DAY_74_SHAPES = tuple(
    MatmulShape(*triple)
    for triple in (
        (1, 1, 1),
        (17, 33, 9),
        (32, 32, 16),
        (31, 47, 129),
        (129, 65, 1000),
        (255, 257, 511),
        (64, 48, 4096),
        (3, 1000, 2048),
        (1000, 999, 100),
    )
)
RAGGED = MatmulShape(37, 45, 29)
TEXTS = {tile: tiled_cuda_source(tile=tile) for tile in TILES}
PINNED = TEXTS[16]
CUBLAS = cublas_source()
SOURCE_SHA256 = {
    8: "3b72d60eb486fb5d1671efcf5428d3b6d93f4c61d0e6ec1b4c97815be31884fa",
    16: "a789a301b41dde5d8449860297a0c5340fabfbc9c8c39209fa15791b038fc867",
    32: "178e6d6ec2976d70ef02e0f9cc38eedbdacdd863588b3e441fecc393826b7d6e",
    "cublas": "9a86b34a84f3550412c2971a11b3972f6d1595b67193812afdb9583e1c36f889",
}
PINNED_CALL = SgemmCall(
    "CUBLAS_OP_N", "CUBLAS_OP_N", "n", "m", "k", "b", "n", "a", "k", "c", "n"
)


def check(text, shape=RAGGED, tile=16):
    return check_tiled_source(text, tile=tile, shape=shape)


def check_cublas(text, shape=RAGGED):
    return check_cublas_source(text, shape=shape)


def flat(text):
    return " ".join(text.replace("//", " ").split())


def test_pinned_texts_are_stable_and_state_that_nothing_ran():
    for key, text in (*TEXTS.items(), ("cublas", CUBLAS)):
        assert source_sha256(text) == SOURCE_SHA256[key]
        assert sha256(text.encode("ascii")).hexdigest() == SOURCE_SHA256[key]
        assert text.isascii() and text.endswith("}\n")
    for text in TEXTS.values():
        assert "The lab that generated this text did not compile or run it." in text
        assert "It is source for a later run on an NVIDIA GPU." in text
    words = flat(CUBLAS)
    assert "did not compile or link it, and it did not call cuBLAS." in words
    assert (
        "ASSUMPTION from the cuBLAS documentation, to check on the NVIDIA host: "
        "with CUBLAS_DEFAULT_MATH, cublasSgemm uses no TF32 Tensor Core math."
    ) in words
    assert "assumption" in TF32_ASSUMPTION and "NVIDIA host" in TF32_ASSUMPTION
    with pytest.raises(GpuTiledError, match="ASCII"):
        source_sha256("café")


@pytest.mark.parametrize("tile", TILES)
def test_each_text_names_its_tiles_phases_barriers_and_grid(tile):
    lines = TEXTS[tile].splitlines()
    for line in (
        f"__global__ void tiled_matmul_t{tile}_kernel(",
        f"    __shared__ float As[{tile}][{tile}];",
        f"    __shared__ float Bs[{tile}][{tile}];",
        f"    for (int phase = 0; phase * {tile} < k; ++phase) {{",
        "        As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;",
        "        Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;",
        f"        for (int i = 0; i < min({tile}, k - phase * {tile}); ++i) {{",
        "            sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);",
        "    if (row < m && col < n) {",
        f"    dim3 block({tile}, {tile});",
        f"    dim3 grid(cdiv(n, {tile}), cdiv(m, {tile}));",
        f"    tiled_matmul_t{tile}_kernel<<<grid, block>>>(a, b, c, m, n, k);",
    ):
        assert line in lines, line
    assert lines.count("        __syncthreads();") == 2
    assert kernel_name(tile) == f"tiled_matmul_t{tile}_kernel"
    assert launcher_name(tile) == f"launch_tiled_matmul_t{tile}"
    assert shared_memory_bytes(tile) == 2 * tile * tile * 4


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("shape", DAY_74_SHAPES, ids=str)
def test_every_check_passes_and_the_grid_is_the_day_74_launch(tile, shape):
    result = check(TEXTS[tile], shape, tile)

    assert result.passed, result.failures
    assert tuple(name for name, _ in result.checks) == TILED_CHECK_NAMES
    assert result.launch == matmul_launch(shape, (tile, tile), X_COLUMNS)
    assert result.shared_memory_bytes == shared_memory_bytes(tile)
    assert result.to_mapping()["tile"] == tile


def test_the_shared_size_is_read_from_the_text_and_the_grid_from_the_shape():
    sizes = {tile: check(TEXTS[tile], tile=tile).shared_memory_bytes for tile in TILES}

    assert sizes == {8: 512, 16: 2048, 32: 8192}
    assert check(PINNED).to_mapping()["launch"] == {
        "grid": [3, 3, 1],
        "block": [16, 16, 1],
    }
    assert check(PINNED, MatmulShape(1000, 999, 100)).launch == LaunchConfig(
        Dim3(63, 63), Dim3(16, 16)
    )
    # The text of one tile is not the text of another tile.
    assert check(PINNED, tile=8).failures == TILED_CHECK_NAMES
    padded = check(PINNED.replace("float As[16][16];", "float As[16][17];"))
    assert padded.shared_memory_bytes == 2048 + 64
    assert padded.failures == ("tiled_shared_memory",)


def test_formatting_and_comments_do_not_change_the_structure():
    spaced = re.sub(r"([(),;<>+*?:\[\]])", r" \1 ", PINNED)
    commented = PINNED.replace("{\n", "{ /* block */\n").replace(";\n", "; // end\n")

    assert check(spaced).passed
    assert check(commented).passed
    assert check_cublas(re.sub(r"([(),;&])", r" \1 ", CUBLAS)).passed
    assert check_cublas(CUBLAS.replace(";\n", "; /* end */\n")).passed


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    [
        ("return (a + b - 1) / b;", "return a / b;", "cdiv_definition"),
        ("int n, int k) {\n    __shared__", "int n, unsigned int k) {\n    __shared__", "tiled_signature"),
        ("__shared__ float As[16][16];", "__shared__ float As[16][17];", "tiled_shared_memory"),
        ("__shared__ float Bs[16][16];", "__shared__ float Bs[8][16];", "tiled_shared_memory"),
        ("__shared__ float As[16][16];", "__shared__ double As[16][16];", "tiled_shared_memory"),
        ("int tx = threadIdx.x;", "int tx = threadIdx.y;", "tiled_index"),
        ("int row = blockIdx.y * 16 + ty;", "int row = blockIdx.x * 16 + ty;", "tiled_index"),
        ("int col = blockIdx.x * 16 + tx;", "int col = blockIdx.x * 16 + ty;", "tiled_index"),
        ("phase * 16 < k;", "phase * 16 <= k;", "tiled_phase_loop"),
        ("++phase) {", "phase += 2) {", "tiled_phase_loop"),
        ("int a_col = phase * 16 + tx;", "int a_col = phase * 16 + ty;", "tiled_loads"),
        ("(row < m && a_col < k)", "(row < m)", "tiled_loads"),
        ("(b_row < k && col < n)", "(col < n)", "tiled_loads"),
        (": 0.0f;\n        Bs", ": 1.0f;\n        Bs", "tiled_loads"),
        ("a[row * k + a_col]", "a[row * n + a_col]", "tiled_loads"),
        ("b[b_row * n + col]", "b[col * k + b_row]", "tiled_loads"),
        ("        __syncthreads();\n        for", "        for", "tiled_barriers"),
        ("        }\n        __syncthreads();\n    }", "        }\n    }", "tiled_barriers"),
        ("        __syncthreads();\n        for", "        __syncwarp();\n        for", "tiled_barriers"),
        ("i < min(16, k - phase * 16)", "i < 16", "tiled_inner_loop"),
        ("i < min(16, k - phase * 16)", "i < min(16, k)", "tiled_inner_loop"),
        ("int i = 0; i < min", "int i = 1; i < min", "tiled_inner_loop"),
        ("; ++i) {", "; i += 2) {", "tiled_inner_loop"),
        ("sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);", "sum += As[ty][i] * Bs[i][tx];", "tiled_fma_order"),
        ("__fmaf_rn(", "__fmaf_rz(", "tiled_fma_order"),
        ("As[ty][i], Bs[i][tx]", "As[i][ty], Bs[tx][i]", "tiled_fma_order"),
        ("float sum = 0.0f;", "float sum = -0.0f;", "tiled_fma_order"),
        ("if (row < m && col < n) {", "if (row < m) {", "tiled_store_guard"),
        ("if (row < m && col < n) {", "if (row <= m && col < n) {", "tiled_store_guard"),
        ("        c[row * n + col] = sum;\n    }\n}", "    }\n    c[row * n + col] = sum;\n}", "tiled_store_guard"),
        ("c[row * n + col] = sum;", "c[col * m + row] = sum;", "tiled_store_guard"),
        ("dim3 grid(cdiv(n, 16), cdiv(m, 16));", "dim3 grid(n / 16, m / 16);", "tiled_grid"),
        ("cdiv(n, 16), cdiv(m, 16)", "cdiv(m, 16), cdiv(n, 16)", "tiled_grid"),
        ("dim3 block(16, 16);", "dim3 block(16, 8);", "tiled_grid"),
        ("<<<grid, block>>>", "<<<block, grid>>>", "tiled_grid"),
        ("(a, b, c, m, n, k);\n}", "(a, b, c, n, m, k);\n}", "tiled_grid"),
    ],
)  # fmt: skip
def test_each_structural_defect_fails_the_check_that_names_it(old, new, failure):
    assert old in PINNED
    result = check(PINNED.replace(old, new, 1))

    assert failure in result.failures
    assert not result.passed


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("        Bs[ty][tx] =", "        As[tx][ty] = 0.0f;\n        Bs[ty][tx] =", ("tiled_loads",)),
        ("        __syncthreads();\n        for", "        for", ("tiled_barriers",)),
        ("+ tx;\n    float sum", "+ tx;\n    __shared__ float Cs[16][16];\n    float sum", ("tiled_shared_memory",)),
        ("float sum = 0.0f;\n", "float sum = 0.0f;\n    sum = 1.0f;\n", ("tiled_fma_order",)),
        ("], sum);\n", "], sum);\n            sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);\n", ("tiled_fma_order",)),
        ("    }\n    if (row", "    }\n    __syncthreads();\n    if (row", ("tiled_phase_loop", "tiled_barriers")),
        ("        c[row * n + col] = sum;\n", "        __syncthreads();\n        c[row * n + col] = sum;\n", ("tiled_barriers", "tiled_store_guard")),
        ("    float sum = 0.0f;\n", "    float sum = 0.0f;\n    if (row >= m) {\n        return;\n    }\n", ("tiled_index", "tiled_store_guard")),
        ("cdiv(m, 16));\n", "cdiv(m, 16));\n    block.x = 8;\n", ("tiled_grid",)),
        ("block(16, 16);\n    dim3 grid(cdiv(n, 16)", "block(020, 16);\n    dim3 grid(cdiv(n, 020)", ("tiled_grid",)),
    ],
)  # fmt: skip
def test_an_extra_statement_fails_only_the_checks_of_its_part(old, new, failures):
    # 020 is the octal literal 16 for a C compiler, which the check refuses.
    assert PINNED.count(old) == 1
    assert check(PINNED.replace(old, new)).failures == failures


@pytest.mark.parametrize(
    "extra",
    [
        "__device__ int min(int a, int b) { return b; }",
        "#define __fmaf_rn(x, y, z) ((x) * (y) + (z))",
        "#define __syncthreads()",
        "#define As Bs",
    ],
)
def test_code_outside_the_three_functions_fails_every_check(extra):
    # A macro or an overload changes what the pinned statements mean.
    marker = "\n__global__ void tiled_matmul_t16_kernel("
    result = check(PINNED.replace(marker, f"\n{extra}\n{marker}", 1))

    assert result.failures == TILED_CHECK_NAMES
    assert result.launch is None and result.shared_memory_bytes is None


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("        __syncthreads();\n        for", "        // keep \\\n        __syncthreads();\n        for", TILED_CHECK_NAMES),
        ("        __syncthreads();\n        for", "        // keep ??/\n        __syncthreads();\n        for", TILED_CHECK_NAMES),
        ("        __syncthreads();\n        for", "        // __syncthreads(); /*\n        for", ("tiled_barriers",)),
        ("        __syncthreads();\n        for", "        /* __syncthreads(); */\n        for", ("tiled_barriers",)),
    ],
)  # fmt: skip
def test_comments_are_read_as_a_compiler_reads_them(old, new, failures):
    assert PINNED.count(old) == 1
    assert check(PINNED.replace(old, new)).failures == failures


def test_require_names_every_failure_and_inputs_are_validated():
    broken = PINNED.replace("i < min(16, k - phase * 16)", "i < 16").replace(
        "cdiv(m, 16)", "cdiv(m, 15)"
    )

    with pytest.raises(GpuTiledError, match="tiled_inner_loop, tiled_grid"):
        require_tiled_source(broken, tile=16, shape=RAGGED)
    assert require_tiled_source(PINNED, tile=16, shape=RAGGED).passed
    for tile in (4, 64, 0):
        with pytest.raises(GpuTiledError, match="tile must be one of"):
            tiled_cuda_source(tile=tile)
    for tile in (True, 16.0, "16"):
        with pytest.raises(TypeError, match="tile must be an integer"):
            tiled_cuda_source(tile=tile)
    with pytest.raises(TypeError, match="must be a string"):
        check(PINNED.encode("ascii"))
    with pytest.raises(TypeError, match="MatmulShape"):
        check_tiled_source(PINNED, tile=16, shape=(37, 45, 29))
    with pytest.raises(GpuTiledError, match="at most 8192"):
        check(PINNED, MatmulShape(8193, 1, 1))
    with pytest.raises(GpuTiledError, match="at most 8192"):
        check_cublas(CUBLAS, MatmulShape(1, 1, 8193))


@pytest.mark.parametrize("shape", DAY_74_SHAPES, ids=str)
def test_the_cublas_text_passes_and_names_the_transposed_call(shape):
    result = check_cublas(CUBLAS, shape)

    assert result.passed, result.failures
    assert tuple(name for name, _ in result.checks) == CUBLAS_CHECK_NAMES
    assert result.call == PINNED_CALL
    assert result.to_mapping()["call"]["lda"] == "n"
    assert CUBLAS.splitlines().count(CUBLAS_HEADER) == 1


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("#include <cublas_v2.h>\n", "", ("cublas_include",)),
        ("const float alpha = 1.0f;", "const float alpha = 2.0f;", ("cublas_scalars",)),
        ("const float beta = 0.0f;", "const float beta = 1.0f;", ("cublas_scalars",)),
        ("CUBLAS_POINTER_MODE_HOST", "CUBLAS_POINTER_MODE_DEVICE", ("cublas_pointer_mode",)),
        ("handle, CUBLAS_DEFAULT_MATH", "handle, CUBLAS_TF32_TENSOR_OP_MATH", ("cublas_math_mode",)),
        ("handle, CUBLAS_DEFAULT_MATH", "handle, CUBLAS_PEDANTIC_MATH", ("cublas_math_mode",)),
        ("MATH);\n    if (status != CUBLAS", "MATH);\n    if (status == CUBLAS", ("cublas_math_mode",)),
        ("n, m, k, &alpha, b, n, a, k", "m, n, k, &alpha, a, k, b, n", ("cublas_row_major",)),
        ("CUBLAS_OP_N, CUBLAS_OP_N, n, m, k", "CUBLAS_OP_T, CUBLAS_OP_T, n, m, k", ("cublas_row_major",)),
        ("b, n, a, k", "b, k, a, n", ("cublas_row_major",)),
        ("&beta,\n        c, n);", "&beta,\n        c, m);", ("cublas_row_major",)),
        ("&beta,\n        c, n);", "&beta,\n        a, n);", ("cublas_row_major",)),
        ("&alpha, b, n", "&beta, b, n", ("cublas_call", "cublas_row_major")),
        ("return cublasSgemm(", "return cublasDgemm(", ("cublas_call", "cublas_row_major")),
        ("CUBLAS_OP_N, CUBLAS_OP_N, n", "CUBLAS_OP_N, CUBLAS_OP_N, 16", ("cublas_call", "cublas_row_major")),
    ],
)  # fmt: skip
def test_each_cublas_defect_fails_the_check_that_names_it(old, new, failures):
    assert CUBLAS.count(old) == 1
    assert check_cublas(CUBLAS.replace(old, new)).failures == failures


@pytest.mark.parametrize(
    "extra",
    [
        "#define CUBLAS_DEFAULT_MATH CUBLAS_TF32_TENSOR_OP_MATH",
        "#include <cublas.h>",
        "static int unused = 0;",
    ],
)
def test_other_code_around_the_cublas_call_fails_every_check(extra):
    marker = "\ncublasStatus_t sgemm_row_major("
    result = check_cublas(CUBLAS.replace(marker, f"\n{extra}\n{marker}", 1))

    assert result.failures == CUBLAS_CHECK_NAMES
    assert result.call is None


def test_a_second_math_mode_or_an_extra_statement_fails_the_cublas_text():
    second_mode = CUBLAS.replace(
        "    return cublasSgemm(",
        "    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);\n"
        "    return cublasSgemm(",
    )

    assert "cublas_math_mode" in check_cublas(second_mode).failures
    assert "cublas_call" in check_cublas(second_mode).failures
    with pytest.raises(GpuTiledError, match="cublas_row_major"):
        require_cublas_source(CUBLAS.replace("b, n, a, k", "b, k, a, n"), shape=RAGGED)
    assert require_cublas_source(CUBLAS, shape=RAGGED).passed
    with pytest.raises(TypeError, match="must be a string"):
        check_cublas(None)


def test_the_column_major_model_shows_why_the_operands_swap():
    shape = MatmulShape(2, 3, 4)
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = list(range(1, 13))
    product = [
        sum(a[row * 4 + p] * b[p * 3 + col] for p in range(4))
        for row in range(2)
        for col in range(3)
    ]

    assert column_major_sgemm(PINNED_CALL, shape, a, b) == product
    assert computes_row_major(PINNED_CALL)
    # The textbook column-major call writes C^T into c: the row-major bytes
    # of the transpose, not of C.
    textbook = SgemmCall(
        "CUBLAS_OP_T", "CUBLAS_OP_T", "m", "n", "k", "a", "k", "b", "n", "c", "m"
    )
    transposed = [product[row * 3 + col] for col in range(3) for row in range(2)]
    assert column_major_sgemm(textbook, shape, a, b) == transposed
    assert not computes_row_major(textbook)
    # A row-major call with the row-major leading dimensions reads past A.
    naive = SgemmCall(
        "CUBLAS_OP_N", "CUBLAS_OP_N", "m", "n", "k", "a", "k", "b", "n", "c", "n"
    )
    assert not call_fits(naive, shape)
    assert column_major_sgemm(naive, shape, a, b) is None
    for item in DAY_74_SHAPES:
        assert call_fits(PINNED_CALL, item)


def test_the_probes_admit_exactly_one_of_every_call_the_pattern_names():
    operations = ("CUBLAS_OP_N", "CUBLAS_OP_T")
    calls = [
        SgemmCall(transa, transb, *extents, a, lda, b, ldb, c, ldc)
        for transa, transb in itertools.product(operations, repeat=2)
        for *extents, lda, ldb, ldc in itertools.product("mnk", repeat=6)
        for a, b, c in itertools.product("abc", repeat=3)
    ]

    assert len(calls) == 78_732
    assert [call for call in calls if computes_row_major(call)] == [PINNED_CALL]
