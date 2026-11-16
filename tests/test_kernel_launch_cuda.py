import re
from hashlib import sha256

import pytest

from ai_journey.kernel_launch_config import (
    KernelLaunchError,
    LaunchLimitError,
    launch_1d,
    launch_2d,
)
from ai_journey.kernel_launch_cuda import (
    CHECK_NAMES,
    check_cuda_source,
    cuda_source,
    require_cuda_source,
)

EXTENTS = {"n": 1000, "threads_per_block": 256, "width": 40, "height": 30}
PINNED = cuda_source(threads_per_block=256, block=(16, 16))


def check(text, **overrides):
    return check_cuda_source(text, **{**EXTENTS, "block": (16, 16), **overrides})


def test_pinned_source_text_is_stable_and_states_it_was_not_compiled():
    assert sha256(PINNED.encode("ascii")).hexdigest() == (
        "cf14adedc20211d11c978cf0d6af291a447853ec01e4d0dee1ecfd02c8631bcc"
    )
    assert PINNED.isascii() and PINNED.endswith("}\n")
    assert "did not compile or run it" in PINNED
    for line in (
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (i < n) {",
        "    int col = blockIdx.x * blockDim.x + threadIdx.x;",
        "    int row = blockIdx.y * blockDim.y + threadIdx.y;",
        "    if (col < width && row < height) {",
        "    dim3 grid(cdiv(n, 256));",
        "    dim3 grid(cdiv(width, 16), cdiv(height, 16));",
    ):
        assert line in PINNED.splitlines(), line


def test_generated_source_passes_every_check_and_agrees_with_the_simulator():
    for vector, block, extents in (
        (256, (16, 16), EXTENTS),
        (128, (32, 8), {"n": 1024, "width": 48, "height": 32}),
        (1024, (1, 1024), {"n": 1, "width": 1, "height": 2049}),
    ):
        text = cuda_source(threads_per_block=vector, block=block)
        values = {**EXTENTS, **extents, "threads_per_block": vector}
        result = check_cuda_source(text, block=block, **values)

        assert result.passed, result.failures
        assert tuple(name for name, _ in result.checks) == CHECK_NAMES
        assert result.vector_add_launch == launch_1d(values["n"], vector)
        assert result.rgb_to_grayscale_launch == launch_2d(
            values["width"], values["height"], *block
        )
    assert check(PINNED).to_mapping()["launches"] == {
        "vector_add": {"grid": [4, 1, 1], "block": [256, 1, 1]},
        "rgb_to_grayscale": {"grid": [3, 2, 1], "block": [16, 16, 1]},
    }


def test_formatting_and_comments_do_not_change_the_structure():
    spaced = re.sub(r"([(),;<>+*])", r" \1 ", PINNED)
    commented = PINNED.replace("{\n", "{ /* block */\n").replace(";\n", "; // end\n")

    assert check(spaced).passed
    assert check(commented).passed


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    [
        ("    if (i < n) {\n        out[i] = a[i] + b[i];\n    }", "    out[i] = a[i] + b[i];", "vector_add_guard"),
        ("out[i] = a[i] + b[i];\n    }", "}\n        out[i] = a[i] + b[i];", "vector_add_guard"),
        ("int i = blockIdx.x * blockDim.x + threadIdx.x;", "int i = blockIdx.x + threadIdx.x;", "vector_add_index"),
        ("const float* a, const float* b, float* out, int n) {\n    int i", "const float* a, const float* b, float* out, unsigned int n) {\n    int i", "vector_add_signature"),
        ("int row = blockIdx.y * blockDim.y + threadIdx.y;", "int row = blockIdx.y * blockDim.x + threadIdx.y;", "rgb_to_grayscale_index"),
        ("if (col < width && row < height) {", "if (col < width) {", "rgb_to_grayscale_guard"),
        ("unsigned int g = rgb[3 * pixel + 1];", "unsigned int g = rgb[3 * pixel + 2];", "rgb_to_grayscale_guard"),
        ("2989u", "2990u", "rgb_to_grayscale_rule"),
        (" + 5000u) / 10000u", ") / 10000u", "rgb_to_grayscale_rule"),
        ("luma < 255u ? luma : 255u", "luma", "rgb_to_grayscale_rule"),
        ("return (a + b - 1) / b;", "return a / b;", "cdiv_definition"),
        ("dim3 grid(cdiv(n, 256));", "dim3 grid(n / 256);", "vector_add_grid"),
        ("dim3 grid(cdiv(n, 256));", "dim3 grid(cdiv(n, 128));", "vector_add_grid"),
        ("dim3 block(256);", "dim3 block(128);", "vector_add_grid"),
        ("cdiv(width, 16), cdiv(height, 16)", "cdiv(height, 16), cdiv(width, 16)", "rgb_to_grayscale_grid"),
        ("dim3 block(16, 16);", "dim3 block(16, 16);\n    dim3 block(8, 8);", "rgb_to_grayscale_grid"),
        ("<<<grid, block>>>(rgb, gray, width, height)", "<<<block, grid>>>(rgb, gray, width, height)", "rgb_to_grayscale_grid"),
    ],
)  # fmt: skip
def test_each_structural_defect_fails_the_check_that_names_it(old, new, failure):
    assert old in PINNED
    result = check(PINNED.replace(old, new, 1))

    assert failure in result.failures
    assert not result.passed


@pytest.mark.parametrize(
    ("old", "new", "failure"),
    [
        ("threadIdx.x;\n    if (i < n)", "threadIdx.x;\n    i += 1;\n    if (i < n)", "vector_add_index"),
        ("threadIdx.y;\n    if (col", "threadIdx.y;\n    col += 1;\n    if (col", "rgb_to_grayscale_index"),
        ("+ col;\n", "+ col;\n        pixel += width;\n", "rgb_to_grayscale_guard"),
        ("/ 10000u;\n", "/ 10000u;\n        luma = 0u;\n", "rgb_to_grayscale_rule"),
        ("255u);\n", "255u);\n        gray[pixel + 1] = 0;\n", "rgb_to_grayscale_rule"),
        ("cdiv(n, 256));\n", "cdiv(n, 256));\n    block.x = 128;\n", "vector_add_grid"),
        ("cdiv(n, 256));\n", "cdiv(n, 256));\n    n = n + 1;\n", "vector_add_grid"),
        ("(a, b, out, n);\n", "(a, b, out, n);\n    vector_add_kernel<<<1, 1>>>(a, b, out, n);\n", "vector_add_grid"),
        ("block(256);\n    dim3 grid(cdiv(n, 256))", "block(0256);\n    dim3 grid(cdiv(n, 0256))", "vector_add_grid"),
        ("cdiv(height, 16));\n", "cdiv(height, 16));\n    block.y = 8;\n", "rgb_to_grayscale_grid"),
    ],
)  # fmt: skip
def test_an_extra_statement_fails_the_checks_of_its_function(old, new, failure):
    # Each edit keeps every statement that the checks look for. 0256 is the
    # octal literal 174 for a C compiler.
    assert PINNED.count(old) == 1
    result = check(PINNED.replace(old, new))

    assert failure in result.failures
    function = failure.rsplit("_", 1)[0]
    assert all(name.startswith(function) for name in result.failures)


@pytest.mark.parametrize(
    "extra",
    [
        "__host__ int cdiv(int a, int b) { return a / b; }",
        "#define threadIdx blockIdx",
    ],
)
def test_code_outside_the_five_functions_fails_every_check(extra):
    # The int overload is the better match for cdiv(n, 256), and the macro
    # rewrites every threadIdx, so the check can confirm no function.
    marker = "\n__global__ void vector_add_kernel("
    result = check(PINNED.replace(marker, f"\n{extra}\n{marker}", 1))

    assert result.failures == CHECK_NAMES
    assert result.vector_add_launch is None
    assert result.rgb_to_grayscale_launch is None


def test_a_second_kernel_definition_breaks_the_unique_signature():
    start = PINNED.index("__global__ void vector_add_kernel(")
    end = PINNED.index("// Exact rule")
    doubled = PINNED[:end] + PINNED[start:end] + PINNED[end:]

    assert "vector_add_signature" in check(doubled).failures


def test_grid_checks_use_the_given_extents():
    # The text is independent of extents; the simulator's grid is not.
    assert check(PINNED, n=1024, width=48, height=32).passed
    assert check(PINNED, threads_per_block=128).failures == ("vector_add_grid",)
    assert check(PINNED, block=(8, 8)).failures == ("rgb_to_grayscale_grid",)


def test_require_names_every_failure_and_source_rejects_bad_blocks():
    broken = PINNED.replace("2989u", "2990u").replace("cdiv(n, 256)", "cdiv(n, 255)")

    with pytest.raises(
        KernelLaunchError, match="vector_add_grid, rgb_to_grayscale_rule"
    ):
        require_cuda_source(broken, block=(16, 16), **EXTENTS)
    assert require_cuda_source(PINNED, block=(16, 16), **EXTENTS).passed
    with pytest.raises(LaunchLimitError, match="max_threads_per_block"):
        cuda_source(threads_per_block=256, block=(64, 32))
    with pytest.raises(LaunchLimitError, match="max_block_dim_x"):
        cuda_source(threads_per_block=2048, block=(16, 16))
    with pytest.raises(KernelLaunchError, match="block_x, block_y"):
        cuda_source(threads_per_block=256, block=[16, 16])
    with pytest.raises(TypeError, match="must be a string"):
        check(PINNED.encode("ascii"))


def test_overlong_dim3_literals_fail_the_grid_check_instead_of_raising():
    # 5,000 digits exceed CPython's 4,300-digit limit for int() on a string.
    huge = "9" * 5_000
    vector = check(PINNED.replace("dim3 block(256);", f"dim3 block({huge});"))
    gray = check(PINNED.replace("cdiv(height, 16)", f"cdiv(height, {huge})"))
    wide = check(PINNED.replace("dim3 block(256);", "dim3 block(99999999999);"))

    assert vector.failures == ("vector_add_grid",)
    assert gray.failures == ("rgb_to_grayscale_grid",)
    assert wide.failures == ("vector_add_grid",)
