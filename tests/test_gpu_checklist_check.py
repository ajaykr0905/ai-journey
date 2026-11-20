"""CPU tests of the compile-free checker. No test compiles or runs CUDA."""

import json
import re
from pathlib import Path

import pytest

from ai_journey.gpu_checklist_check import (
    BENCHMARK_SHAPES,
    BLOCKED_CHECK_NAMES,
    CHECK_SHAPES,
    COARSENED_CHECK_NAMES,
    CORRECTNESS_SHAPES,
    Node,
    check_cuda_source,
    check_names,
    parse_body,
    parse_kernel,
    require_cuda_source,
    shared_hazards,
)
from ai_journey.gpu_checklist_cuda import (
    PINNED_BLOCKED,
    PINNED_COARSENED,
    PINNED_KERNELS,
    GpuChecklistError,
    cuda_source,
)
from ai_journey.kernel_launch_config import Dim3, LaunchConfig
from ai_journey.memory_access_tiling import MatmulShape

ROOT = Path(__file__).resolve().parents[1]
RAGGED = MatmulShape(37, 45, 29)
COARSENED = PINNED_COARSENED[1]
BLOCKED = PINNED_BLOCKED[0]
COARSENED_TEXT = cuda_source(COARSENED)
BLOCKED_TEXT = cuda_source(BLOCKED)


def check(text, kernel=COARSENED, shape=RAGGED):
    return check_cuda_source(text, kernel=kernel, shape=shape)


def mutate(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_the_shapes_are_the_day_74_correctness_and_benchmark_shapes():
    config = json.loads((ROOT / "config/day-74-matmul-metal.json").read_bytes())

    assert [[s.m, s.n, s.k] for s in CORRECTNESS_SHAPES] == config["correctness"][
        "shapes"
    ]
    assert [[s.m, s.n, s.k] for s in BENCHMARK_SHAPES] == config["benchmark"]["shapes"]
    assert CHECK_SHAPES == (*CORRECTNESS_SHAPES, *BENCHMARK_SHAPES)
    assert len(set(CHECK_SHAPES)) == 18


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
def test_every_pinned_text_passes_every_check_for_every_shape(kernel):
    text = cuda_source(kernel)
    for shape in CHECK_SHAPES:
        result = check(text, kernel, shape)

        assert result.passed, (shape, result.failures)
        assert tuple(name for name, _ in result.checks) == check_names(kernel)
        assert result.launch == kernel.launch(shape)
        assert result.shared_memory_bytes == kernel.shared_memory_bytes
    assert shared_hazards(parse_kernel(text, kernel=kernel), ("As", "Bs")) == 0


def test_the_mapping_and_the_names_follow_the_kernel_kind():
    mapping = check(COARSENED_TEXT, shape=MatmulShape(1000, 999, 100)).to_mapping()

    assert mapping == {
        "kernel": "coarsened_matmul_t16_c4",
        "checks": dict.fromkeys(COARSENED_CHECK_NAMES, True),
        "passed": True,
        "shared_memory_bytes": 2048,
        "launch": {"grid": [16, 63, 1], "block": [16, 16, 1]},
    }
    assert check(BLOCKED_TEXT, BLOCKED).launch == LaunchConfig(Dim3(1, 1), Dim3(16, 16))
    assert check_names(BLOCKED) == BLOCKED_CHECK_NAMES
    assert len(COARSENED_CHECK_NAMES) == 11 and len(BLOCKED_CHECK_NAMES) == 12
    # The text of one kernel is not the text of another kernel.
    assert check(COARSENED_TEXT, PINNED_COARSENED[0]).failures == COARSENED_CHECK_NAMES
    assert check(BLOCKED_TEXT, PINNED_BLOCKED[1]).failures == BLOCKED_CHECK_NAMES


@pytest.mark.parametrize(
    ("kernel", "text"), [(COARSENED, COARSENED_TEXT), (BLOCKED, BLOCKED_TEXT)]
)
def test_formatting_and_comments_do_not_change_the_structure(kernel, text):
    # Spaces go around whole tokens: "+ +i" is another text, below.
    spaced = re.sub(r"(\+\+|<<<|>>>|&&|[(),;<>+*?:\[\]{}])", r" \1 ", text)
    commented = text.replace("{\n", "{ /* block */\n").replace(";\n", "; // end\n")

    assert check(spaced, kernel).passed
    assert check(commented, kernel).passed


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("++phase", "+ +phase"),
        ("++i", "+ +i"),
        ("++i", "+/* split */+i"),
        ("<<<grid", "< <<grid"),
        ("row < m && col < n", "row < m & & col < n"),
    ],
)
def test_a_space_inside_an_operator_fails_every_check(old, new):
    # A compiler reads "+ +i" as two unary plus signs and no increment.
    result = check(mutate(COARSENED_TEXT, old, new))

    assert result.failures == COARSENED_CHECK_NAMES
    assert result.launch is None and result.shared_memory_bytes is None


@pytest.mark.parametrize(
    "extra",
    [
        "__device__ int min(int a, int b) { return b; }",
        "#define __fmaf_rn(x, y, z) ((x) * (y) + (z))",
        "#define __syncthreads()",
        "#define As Bs",
    ],
)
@pytest.mark.parametrize(
    ("kernel", "text"), [(COARSENED, COARSENED_TEXT), (BLOCKED, BLOCKED_TEXT)]
)
def test_code_outside_the_three_functions_fails_every_check(kernel, text, extra):
    # A macro or an overload changes what the pinned statements mean.
    marker = f"\n__global__ void {kernel.kernel_name}("
    result = check(mutate(text, marker, f"\n{extra}\n{marker}"), kernel)

    assert result.failures == check_names(kernel)


@pytest.mark.parametrize(
    ("new", "failures"),
    [
        (
            "            // keep \\\n            __syncthreads();\n            for",
            "all",
        ),
        (
            "            // keep ??/\n            __syncthreads();\n            for",
            "all",
        ),
        (
            "            // __syncthreads(); /*\n            for",
            ("coarsened_barriers",),
        ),
        (
            "            /* __syncthreads(); */\n            for",
            ("coarsened_barriers",),
        ),
    ],
)
def test_comments_are_read_as_a_compiler_reads_them(new, failures):
    old = "            __syncthreads();\n            for"
    result = check(mutate(COARSENED_TEXT, old, new))

    assert result.failures == (COARSENED_CHECK_NAMES if failures == "all" else failures)


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("return (a + b - 1) / b;", "return a / b;", ("cdiv_definition",)),
        ("__shared__ float As[16][16];", "__shared__ float As[16][17];", ("coarsened_shared_memory",)),
        ("__shared__ float Bs[16][16];", "__shared__ double Bs[16][16];", ("coarsened_shared_memory",)),
        ("int ty = threadIdx.y;", "int ty = threadIdx.x;", ("coarsened_index",)),
        ("int row = blockIdx.y * 16 + ty;", "int row = blockIdx.x * 16 + ty;", ("coarsened_index",)),
        ("int col0 = blockIdx.x * 64 + tx;", "int col0 = blockIdx.x * 16 + tx;", ("coarsened_index",)),
        ("phase * 16 < k;", "phase * 16 <= k;", ("coarsened_phase_loop",)),
        ("++phase) {", "phase += 2) {", ("coarsened_phase_loop",)),
        ("int a_col = phase * 16 + tx;", "int a_col = phase * 16 + ty;", ("coarsened_loads",)),
        ("(row < m && a_col < k)", "(row < m)", ("coarsened_loads",)),
        ("(b_row < k && col < n)", "(col < n)", ("coarsened_loads",)),
        (": 0.0f;\n        for", ": 1.0f;\n        for", ("coarsened_loads",)),
        ("a[row * k + a_col]", "a[row * n + a_col]", ("coarsened_loads",)),
        ("b[b_row * n + col]", "b[col * k + b_row]", ("coarsened_loads",)),
        ("            int col = col0 + j * 16;\n            Bs", "            int col = col0 + j;\n            Bs", ("coarsened_loads",)),
        ("        for (int j = 0; j < 4; ++j) {\n            int col", "        for (int j = 0; j < 3; ++j) {\n            int col", ("coarsened_loads",)),
        ("            __syncthreads();\n            for", "            for", ("coarsened_barriers",)),
        ("            }\n            __syncthreads();\n        }", "            }\n        }", ("coarsened_barriers",)),
        ("            __syncthreads();\n            for", "            __syncwarp();\n            for", ("coarsened_loads", "coarsened_barriers")),
        ("i < min(16, k - phase * 16)", "i < 16", ("coarsened_inner_loop",)),
        ("i < min(16, k - phase * 16)", "i < min(16, k)", ("coarsened_inner_loop",)),
        ("int i = 0; i < min", "int i = 1; i < min", ("coarsened_inner_loop",)),
        ("; ++i) {", "; i += 2) {", ("coarsened_inner_loop",)),
        ("sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j]);", "sum[j] += As[ty][i] * Bs[i][tx];", ("coarsened_fma_order",)),
        ("__fmaf_rn(", "__fmaf_rz(", ("coarsened_fma_order",)),
        ("As[ty][i], Bs[i][tx]", "As[i][ty], Bs[tx][i]", ("coarsened_fma_order",)),
        ("], sum[j]);", "], sum[0]);", ("coarsened_fma_order",)),
        ("sum[j] = 0.0f;", "sum[j] = -0.0f;", ("coarsened_fma_order",)),
        ("float sum[4];", "float sum[2];", ("coarsened_fma_order",)),
        ("if (row < m && col < n) {", "if (row < m) {", ("coarsened_store_guard",)),
        ("if (row < m && col < n) {", "if (row <= m && col < n) {", ("coarsened_store_guard",)),
        ("c[row * n + col] = sum[j];", "c[col * m + row] = sum[j];", ("coarsened_store_guard",)),
        ("    for (int j = 0; j < 4; ++j) {\n        int col = col0 + j * 16;\n        if", "    for (int j = 0; j < 2; ++j) {\n        int col = col0 + j * 16;\n        if", ("coarsened_store_guard",)),
        ("dim3 grid(cdiv(n, 64), cdiv(m, 16));", "dim3 grid(n / 64, m / 16);", ("coarsened_grid",)),
        ("cdiv(n, 64), cdiv(m, 16)", "cdiv(n, 16), cdiv(m, 16)", ("coarsened_grid",)),
        ("dim3 block(16, 16);", "dim3 block(16, 8);", ("coarsened_grid",)),
        ("<<<grid, block>>>", "<<<block, grid>>>", ("coarsened_grid",)),
        ("(a, b, c, m, n, k);\n}", "(a, b, c, n, m, k);\n}", ("coarsened_grid",)),
    ],
)  # fmt: skip
def test_each_coarsened_defect_fails_the_check_that_names_it(old, new, failures):
    assert check(mutate(COARSENED_TEXT, old, new)).failures == failures


@pytest.mark.parametrize(
    ("old", "new", "failures"),
    [
        ("__shared__ float As[16][64];", "__shared__ float As[64][16];", ("blocked_shared_memory",)),
        ("__shared__ float Bs[16][64];", "__shared__ float Bs[16][65];", ("blocked_shared_memory",)),
        ("int t = ty * 16 + tx;", "int t = tx * 16 + ty;", ("blocked_index",)),
        ("int row0 = blockIdx.y * 64 + ty * 4;", "int row0 = blockIdx.y * 64 + ty;", ("blocked_index",)),
        ("int a_c = t % 4 * 4;", "int a_c = t % 4 * 2;", ("blocked_index", "blocked_vector_alignment")),
        ("int b_col = blockIdx.x * 64 + b_c;", "int b_col = blockIdx.x * 64 + b_c + 1;", ("blocked_index", "blocked_vector_alignment")),
        ("n % 4 == 0 &&", "n % 2 == 0 &&", ("blocked_vector_alignment",)),
        ("(a) % 16 == 0", "(a) % 8 == 0", ("blocked_vector_alignment",)),
        ("bool b_vector =\n        n % 4 == 0 && reinterpret_cast<unsigned long long>(b) % 16 == 0;", "bool b_vector = true;", ("blocked_vector_alignment",)),
        ("if (b_vector) {", "if (true) {", ("blocked_vector_alignment", "blocked_loads")),
        ("(&a[a_row * k + a_col])", "(&a[a_row * k + a_col + 1])", ("blocked_vector_alignment", "blocked_loads")),
        ("(a_row < m && a_col < k)\n", "(a_row < m)\n", ("blocked_vector_alignment", "blocked_loads")),
        ("int a_col = phase * 16 + a_c;", "int a_col = phase * 8 + a_c;", ("blocked_vector_alignment", "blocked_loads")),
        ("(a_row < m && a_col + q < k)", "(a_row < m && a_col < k)", ("blocked_loads",)),
        ("As[a_c + 3][a_r] = v.w;", "As[a_c + 3][a_r] = v.z;", ("blocked_loads",)),
        ("b[b_row * n + b_col + q] : 0.0f;", "b[b_row * n + b_col + q] : 1.0f;", ("blocked_loads",)),
        ("        __syncthreads();\n        for", "        for", ("blocked_barriers",)),
        ("        }\n        __syncthreads();\n    }", "        }\n    }", ("blocked_barriers",)),
        ("i < min(16, k - phase * 16)", "i < 16", ("blocked_inner_loop",)),
        ("a_reg[mi] = As[i][ty * 4 + mi];", "a_reg[mi] = As[i][tx * 4 + mi];", ("blocked_fma_order",)),
        ("__fmaf_rn(a_reg[mi], b_reg[ni],", "__fmaf_rn(a_reg[mi], b_reg[mi],", ("blocked_fma_order",)),
        ("sum[mi][ni] = 0.0f;", "sum[mi][ni] = -0.0f;", ("blocked_fma_order",)),
        ("            for (int mi = 0; mi < 4; ++mi) {\n                a_reg", "            for (int mi = 0; mi < 2; ++mi) {\n                a_reg", ("blocked_fma_order",)),
        ("float b_reg[4];", "float b_reg[8];", ("blocked_fma_order",)),
        ("if (row < m && col < n) {", "if (row < m) {", ("blocked_store_guard",)),
        ("int col = col0 + ni;", "int col = col0 + ni * 16;", ("blocked_store_guard",)),
        ("c[row * n + col] = sum[mi][ni];", "c[row * n + col] = sum[ni][mi];", ("blocked_store_guard",)),
        ("cdiv(m, 64));", "cdiv(m, 16));", ("blocked_grid",)),
        ("dim3 block(16, 16);", "dim3 block(16, 8);", ("blocked_grid",)),
    ],
)  # fmt: skip
def test_each_blocked_defect_fails_the_check_that_names_it(old, new, failures):
    assert check(mutate(BLOCKED_TEXT, old, new), BLOCKED).failures == failures


@pytest.mark.parametrize(
    ("kernel", "old", "new", "failures"),
    [
        (COARSENED, "            Bs[ty][tx] =", "            As[tx][ty] = 0.0f;\n            Bs[ty][tx] =", ("coarsened_loads",)),
        (COARSENED, "        As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;\n        for (int j = 0; j < 4; ++j) {\n            int col = col0 + j * 16;\n", "        for (int j = 0; j < 4; ++j) {\n            int col = col0 + j * 16;\n            As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;\n", ("coarsened_loads",)),
        (COARSENED, "+ tx;\n    float sum", "+ tx;\n    __shared__ float Cs[16][16];\n    float sum", ("coarsened_shared_memory",)),
        (COARSENED, "float sum[4];\n", "float sum[4];\n    sum[0] = 1.0f;\n", ("coarsened_fma_order",)),
        (COARSENED, "], sum[j]);\n", "], sum[j]);\n                sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j]);\n", ("coarsened_fma_order",)),
        (COARSENED, "    }\n    for (int j = 0; j < 4; ++j) {\n        int col = col0 + j * 16;\n        if", "    }\n    __syncthreads();\n    for (int j = 0; j < 4; ++j) {\n        int col = col0 + j * 16;\n        if", ("coarsened_phase_loop", "coarsened_barriers")),
        (COARSENED, "            c[row * n + col] = sum[j];\n", "            __syncthreads();\n            c[row * n + col] = sum[j];\n", ("coarsened_barriers", "coarsened_store_guard")),
        (COARSENED, "    float sum[4];\n", "    float sum[4];\n    if (row >= m) {\n        return;\n    }\n", ("coarsened_phase_loop", "coarsened_barriers", "coarsened_store_guard")),
        (COARSENED, "for (int phase = 0; phase * 16 < k; ++phase)", "for (int phase = 0; phase * 16 < k - tx; ++phase)", ("coarsened_phase_loop", "coarsened_barriers")),
        (COARSENED, "cdiv(m, 16));\n", "cdiv(m, 16));\n    block.x = 8;\n", ("coarsened_grid",)),
        (COARSENED, "block(16, 16);\n", "block(020, 16);\n", ("coarsened_grid",)),
        (BLOCKED, "            As[a_c][a_r] = v.x;", "            __syncthreads();\n            As[a_c][a_r] = v.x;", ("blocked_loads", "blocked_barriers")),
        (BLOCKED, "        __syncthreads();\n        for (int i", "        __syncthreads();\n        Bs[0][0] = 0.0f;\n        for (int i", ("blocked_loads", "blocked_barriers")),
        (BLOCKED, "        if (b_vector) {", "        if (tx < 4) {\n            __syncthreads();\n        }\n        if (b_vector) {", ("blocked_loads", "blocked_barriers")),
        (BLOCKED, "    float sum[4][4];\n", "    __shared__ float Cs[4][4];\n    float sum[4][4];\n", ("blocked_shared_memory",)),
    ],
)  # fmt: skip
def test_an_extra_statement_fails_only_the_checks_of_its_part(
    kernel, old, new, failures
):
    # 020 is the octal literal 16 for a C compiler, which the check refuses.
    text = cuda_source(kernel)

    assert check(mutate(text, old, new), kernel).failures == failures


def test_the_hazard_scan_names_the_race_that_each_missing_barrier_leaves():
    first = "            __syncthreads();\n            for"
    second = "            }\n            __syncthreads();\n        }"
    no_first = mutate(COARSENED_TEXT, first, "            for")
    no_second = mutate(COARSENED_TEXT, second, "            }\n        }")
    tiles = ("As", "Bs")

    def hazards(text):
        return shared_hazards(parse_kernel(text, kernel=COARSENED), tiles)

    assert hazards(COARSENED_TEXT) == 0
    # Without the first barrier, the reads of each B tile meet its write.
    assert hazards(no_first) > 0
    # Without the second, the next write of Bs, and of As in the next
    # phase, meets the reads before it.
    assert hazards(no_second) > 0
    assert parse_kernel("int x;", kernel=COARSENED) is None


@pytest.mark.parametrize(
    "code",
    [
        "for(int i=0;i<4;++i)x=1;",
        "while(x){y=1;}",
        "if(x){y=1;}else if(z){w=1;}",
        "do{x=1;}while(0);",
        "float s[2]={0.0f,0.0f};",
        "x=1",
    ],
)
def test_the_parser_refuses_other_blocks(code):
    assert parse_body(code) is None


def test_the_parser_keeps_loops_guards_and_else_blocks():
    nodes = parse_body("int a=0;for(int i=0;i<2;++i){if(a<1){a=1;}else{a=2;}}")

    assert nodes == (
        Node("statement", "int a=0;"),
        Node(
            "for",
            "for(int i=0;i<2;++i)",
            (
                Node(
                    "if",
                    "if(a<1)",
                    (Node("statement", "a=1;"),),
                    (Node("statement", "a=2;"),),
                ),
            ),
        ),
    )


def test_require_names_every_failure_and_inputs_are_validated():
    broken = COARSENED_TEXT.replace("i < min(16, k - phase * 16)", "i < 16").replace(
        "cdiv(m, 16)", "cdiv(m, 15)"
    )

    with pytest.raises(GpuChecklistError, match="coarsened_inner_loop, coarsened_grid"):
        require_cuda_source(broken, kernel=COARSENED, shape=RAGGED)
    assert require_cuda_source(COARSENED_TEXT, kernel=COARSENED, shape=RAGGED).passed
    with pytest.raises(TypeError, match="must be a string"):
        check(COARSENED_TEXT.encode("ascii"))
    with pytest.raises(TypeError, match="CoarsenedKernel or BlockedKernel"):
        check_cuda_source(COARSENED_TEXT, kernel=(16, 4), shape=RAGGED)
    with pytest.raises(TypeError, match="MatmulShape"):
        check_cuda_source(COARSENED_TEXT, kernel=COARSENED, shape=(37, 45, 29))
    with pytest.raises(GpuChecklistError, match="at most 8192"):
        check(COARSENED_TEXT, shape=MatmulShape(1, 8193, 1))
