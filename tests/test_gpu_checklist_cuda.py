"""CPU tests of the pinned Day 77 CUDA texts. No test compiles or runs CUDA."""

import re
from hashlib import sha256

import pytest

from ai_journey.gpu_checklist_cuda import (
    CUBLAS_FILE,
    PINNED_BLOCKED,
    PINNED_COARSENED,
    PINNED_KERNELS,
    BlockedKernel,
    CoarsenedKernel,
    GpuChecklistError,
    blocked_cuda_source,
    coarsened_cuda_source,
    cuda_source,
    require_kernel,
    require_shape,
    source_files,
    source_sha256,
)
from ai_journey.gpu_tiled_config import PINNED_PLAN
from ai_journey.gpu_tiled_cuda import GpuTiledError, cublas_source
from ai_journey.kernel_launch_config import Dim3, LaunchConfig
from ai_journey.memory_access_tiling import MatmulShape

SOURCE_SHA256 = {
    "coarsened_matmul_t16_c2": (
        "5a57f3c70b5e10cd958d6b5b23e804cee6fe91bf5294ff87ae3cbc9b6b1d6ab7"
    ),
    "coarsened_matmul_t16_c4": (
        "a4a6110ff1be048cfa79490bf185a0a725a05249bc68a6aad482cc662c8bf74c"
    ),
    "coarsened_matmul_t32_c2": (
        "41e1bbc8274250492ab9316d6721822a982baa5a6cebac126c17cfd1e7494550"
    ),
    "blocked_matmul_m64_n64_k16_t4x4": (
        "71fb4757e0e4f52fc4d59dfebb42a6863fc3fb2dc4f6215cd058060b795d2081"
    ),
    "blocked_matmul_m128_n128_k8_t8x8": (
        "b8438783e768956b556e375380ebd97bd0123ce32cec6e773cee41dfbc8eee0c"
    ),
}
# The Day 76 cuBLAS text, unchanged.
CUBLAS_SHA256 = "9a86b34a84f3550412c2971a11b3972f6d1595b67193812afdb9583e1c36f889"


def lines_of(kernel):
    return cuda_source(kernel).splitlines()


def test_pinned_texts_are_stable_and_state_that_nothing_ran():
    assert [kernel.name for kernel in PINNED_KERNELS] == list(SOURCE_SHA256)
    for kernel in PINNED_KERNELS:
        text = cuda_source(kernel)
        assert source_sha256(text) == SOURCE_SHA256[kernel.name]
        assert sha256(text.encode("ascii")).hexdigest() == SOURCE_SHA256[kernel.name]
        assert text.isascii() and text.endswith("}\n")
        assert "The lab that generated this text did not compile or run it." in text
        assert "It is source for a later run on an NVIDIA GPU." in text
        assert "#" not in text and "\\" not in text and "??" not in text
    with pytest.raises(GpuChecklistError, match="ASCII"):
        source_sha256("café")


@pytest.mark.parametrize("kernel", PINNED_COARSENED, ids=lambda item: item.name)
def test_each_coarsened_text_loads_a_once_and_reuses_it_for_each_b_tile(kernel):
    t, factor, width = kernel.tile, kernel.coarsening, kernel.width
    lines = lines_of(kernel)
    expected = [
        f"__global__ void coarsened_matmul_t{t}_c{factor}_kernel(",
        f"    __shared__ float As[{t}][{t}];",
        f"    __shared__ float Bs[{t}][{t}];",
        f"    int row = blockIdx.y * {t} + ty;",
        f"    int col0 = blockIdx.x * {width} + tx;",
        f"    float sum[{factor}];",
        f"    for (int phase = 0; phase * {t} < k; ++phase) {{",
        "        As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;",
        f"        for (int j = 0; j < {factor}; ++j) {{",
        f"            int col = col0 + j * {t};",
        "            Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;",
        f"            for (int i = 0; i < min({t}, k - phase * {t}); ++i) {{",
        "                sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j]);",
        "        if (row < m && col < n) {",
        "            c[row * n + col] = sum[j];",
        f"    dim3 block({t}, {t});",
        f"    dim3 grid(cdiv(n, {width}), cdiv(m, {t}));",
        f"    coarsened_matmul_t{t}_c{factor}_kernel<<<grid, block>>>(a, b, c, m, n, k);",
    ]
    for line in expected:
        assert line in lines, line
    assert lines.count("            __syncthreads();") == 2
    # The A tile is loaded before the loop over the B tiles, once per phase.
    assert lines.index(expected[7]) < lines.index(expected[8])
    assert kernel.block == (t, t) and kernel.threads_per_block == t * t
    assert kernel.shared_memory_bytes == 2 * t * t * 4
    assert kernel.kernel_name == f"{kernel.name}_kernel"
    assert kernel.launcher_name == f"launch_{kernel.name}"


@pytest.mark.parametrize("kernel", PINNED_BLOCKED, ids=lambda item: item.name)
def test_each_blocked_text_stages_one_group_of_a_and_b_per_thread(kernel):
    bm, bn, bk = kernel.block_m, kernel.block_n, kernel.block_k
    tm, tn = kernel.thread_m, kernel.thread_n
    x, y = kernel.block
    lines = lines_of(kernel)
    for line in (
        f"    __shared__ float As[{bk}][{bm}];",
        f"    __shared__ float Bs[{bk}][{bn}];",
        f"    int t = ty * {x} + tx;",
        f"    int a_r = t / {bk // 4};",
        f"    int a_c = t % {bk // 4} * 4;",
        f"    int b_r = t / {bn // 4};",
        f"    int b_c = t % {bn // 4} * 4;",
        "        k % 4 == 0 && reinterpret_cast<unsigned long long>(a) % 16 == 0;",
        "        n % 4 == 0 && reinterpret_cast<unsigned long long>(b) % 16 == 0;",
        f"    float sum[{tm}][{tn}];",
        f"            for (int mi = 0; mi < {tm}; ++mi) {{",
        f"                a_reg[mi] = As[i][ty * {tm} + mi];",
        f"                b_reg[ni] = Bs[i][tx * {tn} + ni];",
        "                    sum[mi][ni] = __fmaf_rn(a_reg[mi], b_reg[ni], sum[mi][ni]);",
        "            if (row < m && col < n) {",
        "                c[row * n + col] = sum[mi][ni];",
        f"    dim3 block({x}, {y});",
        f"    dim3 grid(cdiv(n, {bn}), cdiv(m, {bm}));",
    ):
        assert line in lines, line
    assert lines.count("        __syncthreads();") == 2
    # Each float4 load sits in the branch of its alignment flag, and the
    # other branch loads 4 guarded scalars.
    text = cuda_source(kernel)
    for name, flag in (("a", "a_vector"), ("b", "b_vector")):
        branch = re.search(
            rf"if \({flag}\) \{{(.*?)\}} else \{{(.*?)\n        \}}", text, re.S
        )
        assert branch is not None
        assert f"reinterpret_cast<const float4*>(&{name}[" in branch.group(1)
        assert "float4" not in branch.group(2)
        assert "for (int q = 0; q < 4; ++q)" in branch.group(2)
    assert text.count("reinterpret_cast<const float4*>") == 2
    assert kernel.threads_per_block == 256 == x * y
    assert kernel.shared_memory_bytes == (bk * bm + bk * bn) * 4 == 8192


def test_the_launch_covers_c_with_the_fewest_blocks():
    shape = MatmulShape(1000, 999, 100)
    t16_c4, t32_c2 = PINNED_COARSENED[1], PINNED_COARSENED[2]
    m64, m128 = PINNED_BLOCKED

    assert t16_c4.launch(shape) == LaunchConfig(Dim3(16, 63), Dim3(16, 16))
    assert t32_c2.launch(shape) == LaunchConfig(Dim3(16, 32), Dim3(32, 32))
    assert m64.launch(shape) == LaunchConfig(Dim3(16, 16), Dim3(16, 16))
    assert m128.launch(shape) == LaunchConfig(Dim3(8, 8), Dim3(16, 16))
    for kernel in PINNED_KERNELS:
        columns, rows = kernel.to_mapping()["output_tile"][::-1]
        for case in PINNED_PLAN.shapes:
            grid = kernel.launch(case).grid
            assert (grid.x - 1) * columns < case.n <= grid.x * columns
            assert (grid.y - 1) * rows < case.m <= grid.y * rows
            assert grid.z == 1


def test_the_mappings_name_each_kernel_and_its_tiles():
    assert PINNED_COARSENED[0].to_mapping() == {
        "kind": "coarsened",
        "name": "coarsened_matmul_t16_c2",
        "tile": 16,
        "coarsening": 2,
        "block": [16, 16],
        "output_tile": [16, 32],
        "threads_per_block": 256,
        "shared_memory_bytes": 2048,
    }
    assert PINNED_BLOCKED[1].to_mapping() == {
        "kind": "blocked",
        "name": "blocked_matmul_m128_n128_k8_t8x8",
        "block_m": 128,
        "block_n": 128,
        "block_k": 8,
        "thread_m": 8,
        "thread_n": 8,
        "block": [16, 16],
        "output_tile": [128, 128],
        "threads_per_block": 256,
        "shared_memory_bytes": 8192,
    }
    assert [kernel.outputs_per_thread for kernel in PINNED_KERNELS] == [2, 4, 2, 16, 64]


def test_the_source_files_end_with_the_day_76_cublas_call():
    files = source_files()

    assert [name for name, _ in files] == [
        *(f"{name}.cu" for name in SOURCE_SHA256),
        CUBLAS_FILE,
    ]
    assert files[-1] == ("sgemm_row_major.cu", cublas_source())
    assert source_sha256(files[-1][1]) == CUBLAS_SHA256
    for (name, text), kernel in zip(files, PINNED_KERNELS):
        assert text == cuda_source(kernel)
        assert name == f"{kernel.name}.cu"


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"tile": 4, "coarsening": 2}, GpuChecklistError, "tile must be one of"),
        ({"tile": 16, "coarsening": 3}, GpuChecklistError, "coarsening must be one of"),
        ({"tile": 16, "coarsening": 1}, GpuChecklistError, "coarsening must be one of"),
        ({"tile": True, "coarsening": 2}, TypeError, "tile must be an integer"),
        ({"tile": 16, "coarsening": 2.0}, TypeError, "coarsening must be an integer"),
    ],
)
def test_coarsened_kernels_refuse_unsupported_tiles(arguments, error, message):
    with pytest.raises(error, match=message):
        CoarsenedKernel(**arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((64, 64, 16, 4, 3), GpuChecklistError, "thread_n must be one of"),
        ((64, 64, 16, 1, 4), GpuChecklistError, "thread_m must be one of"),
        ((60, 64, 16, 8, 4), GpuChecklistError, "micro-tile must divide"),
        ((32, 32, 16, 2, 2), GpuChecklistError, "one group of 4 floats"),
        ((64, 64, 8, 4, 4), GpuChecklistError, "one group of 4 floats"),
        ((8, 8, 16, 4, 4), GpuChecklistError, "whole warps"),
        ((256, 256, 4, 2, 2), GpuChecklistError, "at most 1024 threads"),
        ((64, 64, 18, 4, 4), GpuChecklistError, "multiples of 4"),
        ((512, 64, 16, 4, 4), GpuChecklistError, "between 4 and 256"),
        ((64, 64, True, 4, 4), TypeError, "block_k must be an integer"),
    ],
)
def test_blocked_kernels_refuse_tiles_that_break_the_staging_rule(
    arguments, error, message
):
    with pytest.raises(error, match=message):
        BlockedKernel(*arguments)


def test_the_text_functions_refuse_other_kernels_and_shapes():
    with pytest.raises(TypeError, match="CoarsenedKernel"):
        coarsened_cuda_source(PINNED_BLOCKED[0])
    with pytest.raises(TypeError, match="BlockedKernel"):
        blocked_cuda_source(PINNED_COARSENED[0])
    with pytest.raises(TypeError, match="CoarsenedKernel or BlockedKernel"):
        require_kernel((16, 2))
    with pytest.raises(GpuChecklistError, match="at most 8192") as raised:
        PINNED_BLOCKED[0].launch(MatmulShape(8193, 1, 1))
    assert isinstance(raised.value, GpuTiledError)
    with pytest.raises(TypeError, match="MatmulShape"):
        require_shape((1, 1, 1))
