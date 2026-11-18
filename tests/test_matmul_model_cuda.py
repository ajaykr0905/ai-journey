import re
from hashlib import sha256

import pytest

from ai_journey.kernel_launch_config import Dim3, LaunchConfig, LaunchLimitError
from ai_journey.matmul_model_cuda import (
    CHECK_NAMES,
    check_cuda_source,
    cuda_source,
    require_cuda_source,
)
from ai_journey.matmul_model_kernels import X_COLUMNS, matmul_launch
from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.memory_access_tiling import MatmulShape

RAGGED = MatmulShape(m=37, n=45, k=29)
PINNED = cuda_source(block=(32, 8))
PINNED_SHA256 = "a97031329a75820abe14a8cd5a3a55dbe728bbedd64871a79f8d5e2e7b815fc0"


def check(text, shape=RAGGED, block=(32, 8)):
    return check_cuda_source(text, shape=shape, block=block)


def test_pinned_text_is_stable_and_states_it_was_not_compiled():
    assert sha256(PINNED.encode("ascii")).hexdigest() == PINNED_SHA256
    assert PINNED.isascii() and PINNED.endswith("}\n")
    assert "did not compile or run it" in PINNED
    lines = PINNED.splitlines()
    for line in (
        "    int row = blockIdx.y * blockDim.y + threadIdx.y;",
        "    int col = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (row < m && col < n) {",
        "        for (int i = 0; i < k; ++i) {",
        "            sum = __fadd_rn(sum, __fmul_rn(a[row * k + i], b[i * n + col]));",
        "        c[row * n + col] = sum;",
        "    dim3 grid(cdiv(n, 32), cdiv(m, 8));",
    ):
        assert line in lines, line


@pytest.mark.parametrize(
    ("shape", "block"),
    [(RAGGED, (32, 8)), (MatmulShape(32, 64, 16), (32, 8)), (MatmulShape(1, 3000, 1), (1024, 1)), (MatmulShape(70, 3, 2), (8, 4))],
)  # fmt: skip
def test_generated_text_passes_every_check_and_matches_the_simulator(shape, block):
    result = check(cuda_source(block=block), shape, block)

    assert result.passed, result.failures
    assert tuple(name for name, _ in result.checks) == CHECK_NAMES
    assert result.launch == matmul_launch(shape, block, X_COLUMNS)
    assert check(PINNED).to_mapping()["launch"] == {
        "grid": [2, 5, 1],
        "block": [32, 8, 1],
    }


def test_formatting_and_comments_do_not_change_the_structure():
    spaced = re.sub(r"([(),;<>+*])", r" \1 ", PINNED)
    commented = PINNED.replace("{\n", "{ /* block */\n").replace(";\n", "; // end\n")

    assert check(spaced).passed
    assert check(commented).passed


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    [
        ("return (a + b - 1) / b;", "return a / b;", "cdiv_definition"),
        ("float* c, int m, int n, int k) {\n    int row", "float* c, int m, int n, unsigned int k) {\n    int row", "matmul_signature"),
        ("int row = blockIdx.y * blockDim.y + threadIdx.y;", "int row = blockIdx.x * blockDim.x + threadIdx.x;", "matmul_index"),
        ("int col = blockIdx.x * blockDim.x + threadIdx.x;", "int col = blockIdx.x * blockDim.y + threadIdx.x;", "matmul_index"),
        ("if (row < m && col < n) {", "if (row < m) {", "matmul_guard"),
        ("if (row < m && col < n) {", "if (col < n) {", "matmul_guard"),
        ("if (row < m && col < n) {", "if (row < n && col < m) {", "matmul_guard"),
        ("if (row < m && col < n) {", "if (row <= m && col < n) {", "matmul_guard"),
        ("        c[row * n + col] = sum;\n    }\n}", "    }\n    c[row * n + col] = sum;\n}", "matmul_guard"),
        ("for (int i = 0; i < k; ++i)", "for (int i = 0; i < n; ++i)", "matmul_loop"),
        ("i < k; ++i", "i <= k; ++i", "matmul_loop"),
        ("int i = 0; i < k", "int i = 1; i < k", "matmul_loop"),
        ("++i) {", "i += 2) {", "matmul_loop"),
        ("a[row * k + i]", "a[row * n + i]", "matmul_accumulate"),
        ("b[i * n + col]", "b[i * k + col]", "matmul_accumulate"),
        ("sum = __fadd_rn(sum, __fmul_rn(a[row * k + i], b[i * n + col]));", "sum += a[row * k + i] * b[i * n + col];", "matmul_accumulate"),
        ("__fmul_rn(", "__fmul_rz(", "matmul_accumulate"),
        ("float sum = 0.0f;", "float sum = 1.0f;", "matmul_accumulate"),
        ("c[row * n + col] = sum;", "c[col * m + row] = sum;", "matmul_store"),
        ("c[row * n + col] = sum;", "c[row * k + col] = sum;", "matmul_store"),
        ("dim3 grid(cdiv(n, 32), cdiv(m, 8));", "dim3 grid(n / 32, m / 8);", "matmul_grid"),
        ("cdiv(n, 32), cdiv(m, 8)", "cdiv(m, 32), cdiv(n, 8)", "matmul_grid"),
        ("cdiv(m, 8)", "cdiv(m, 16)", "matmul_grid"),
        ("dim3 block(32, 8);", "dim3 block(8, 32);", "matmul_grid"),
        ("<<<grid, block>>>", "<<<block, grid>>>", "matmul_grid"),
        ("(a, b, c, m, n, k);\n}", "(a, b, c, n, m, k);\n}", "matmul_grid"),
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
        ("threadIdx.y;\n    int col", "threadIdx.y;\n    row += 1;\n    int col", ("matmul_index",)),
        ("float sum = 0.0f;\n", "float sum = 0.0f;\n        sum = 1.0f;\n", ("matmul_accumulate",)),
        ("+ col]));\n", "+ col]));\n            sum = 0.0f;\n", ("matmul_accumulate",)),
        ("        c[row * n + col] = sum;\n", "        sum = -sum;\n        c[row * n + col] = sum;\n", ("matmul_store",)),
        ("        c[row * n + col] = sum;\n", "        c[row * n + col] = sum;\n        c[0] = sum;\n", ("matmul_store",)),
        ("        c[row * n + col] = sum;\n", "        for (int j = 0; j < k; ++j) {\n        }\n        c[row * n + col] = sum;\n", ("matmul_loop", "matmul_accumulate", "matmul_store")),
        ("cdiv(m, 8));\n", "cdiv(m, 8));\n    block.x = 16;\n", ("matmul_grid",)),
        ("block(32, 8);\n    dim3 grid(cdiv(n, 32)", "block(040, 8);\n    dim3 grid(cdiv(n, 040)", ("matmul_grid",)),
        ("(a, b, c, m, n, k);\n}", "(a, b, c, m, n, k);\n    naive_matmul_kernel<<<1, 1>>>(a, b, c, m, n, k);\n}", ("matmul_grid",)),
    ],
)  # fmt: skip
def test_an_extra_statement_fails_only_the_checks_of_its_part(old, new, failures):
    # 040 is the octal literal 32 for a C compiler, which the check refuses.
    assert PINNED.count(old) == 1
    assert check(PINNED.replace(old, new)).failures == failures


@pytest.mark.parametrize(
    "extra",
    [
        "__host__ int cdiv(int a, int b) { return a / b; }",
        "#define threadIdx blockIdx",
        "#define __fadd_rn(x, y) ((x) + (y))",
    ],
)
def test_code_outside_the_three_functions_fails_every_check(extra):
    # The macro that redefines __fadd_rn would let the compiler fuse again.
    marker = "\n__global__ void naive_matmul_kernel("
    result = check(PINNED.replace(marker, f"\n{extra}\n{marker}", 1))

    assert result.failures == CHECK_NAMES
    assert result.launch is None


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("    if (row < m && col < n) {\n", "    // keep the guard \\\n    if (row < m && col < n)\n    {\n", CHECK_NAMES),
        ("    if (row < m && col < n) {\n", "    // keep the guard ??/\n    if (row < m && col < n)\n    {\n", CHECK_NAMES),
        ("threadIdx.y;\n    int col", "threadIdx.y; // note\r    row += 1;\n    int col", ("matmul_index",)),
        ("threadIdx.y;\n    int col", "threadIdx.y; // /*\n    row += 1;\n    // */\n    int col", ("matmul_index",)),
    ],
)  # fmt: skip
def test_comments_are_read_as_a_compiler_reads_them(old, new, failures):
    assert PINNED.count(old) == 1
    assert check(PINNED.replace(old, new)).failures == failures


def test_the_grid_check_uses_the_given_extents_and_block():
    assert check(PINNED, MatmulShape(32, 64, 16)).passed
    assert check(PINNED, block=(32, 16)).failures == ("matmul_grid",)
    assert check(
        PINNED.replace("dim3 block(32, 8);", f"dim3 block({'9' * 5_000}, 8);")
    ).failures == ("matmul_grid",)
    # The text never names k in the launch math, so k does not change it.
    assert check(PINNED, MatmulShape(37, 45, 1)).launch == LaunchConfig(
        Dim3(2, 5), Dim3(32, 8)
    )


def test_require_names_every_failure_and_source_rejects_bad_blocks():
    broken = PINNED.replace("i < k;", "i < n;").replace("cdiv(m, 8)", "cdiv(m, 7)")

    with pytest.raises(MatmulModelError, match="matmul_loop, matmul_grid"):
        require_cuda_source(broken, shape=RAGGED, block=(32, 8))
    assert require_cuda_source(PINNED, shape=RAGGED, block=(32, 8)).passed
    with pytest.raises(LaunchLimitError, match="max_threads_per_block"):
        cuda_source(block=(64, 32))
    with pytest.raises(MatmulModelError, match="block_x, block_y"):
        cuda_source(block=[32, 8])
    with pytest.raises(TypeError, match="must be a string"):
        check(PINNED.encode("ascii"))
    with pytest.raises(TypeError, match="MatmulShape"):
        check_cuda_source(PINNED, shape=(37, 45, 29), block=(32, 8))
