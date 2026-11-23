"""Tests of the pinned Triton text, its declared configs and its tile order.

None of these tests imports triton: they read the text as Python source, run
the grouped order with a stand-in for ``tl``, and check the closed forms.
"""

import ast
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.triton_matmul_kernel import (
    AUTOTUNE_CONFIGS,
    AUTOTUNE_KEY,
    CONFIGS_BY_NAME,
    DOT,
    GROUPED_ORDER,
    KERNEL_NAME,
    LAUNCHER_NAME,
    LOAD_MASKS,
    STORE_MASK,
    TRITON_SOURCE,
    TritonConfig,
    TritonMatmulError,
    k_blocks,
    program_tile,
    programs,
    require_shape,
    source_sha256,
    tile_grid,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA256 = "9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56"
IMPORT_CHECK = """
import sys
import ai_journey
import ai_journey.triton_matmul_kernel
print("triton" in sys.modules, "torch" in sys.modules)
"""


class StubLanguage:
    """The two ``tl`` calls of the grouped order, for one program id."""

    def __init__(self, pid):
        self.pid = pid

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    @staticmethod
    def cdiv(numerator, denominator):
        return -(-numerator // denominator)


def tree():
    return ast.parse(TRITON_SOURCE)


def function(name):
    found = [
        node
        for node in tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(found) == 1
    return found[0]


def calls(node, dotted):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == dotted
    ]


def text_order():
    """Return the grouped order lines as the kernel text holds them."""

    lines = [line.strip() for line in TRITON_SOURCE.splitlines()]
    start = lines.index(GROUPED_ORDER[0])
    return tuple(lines[start : start + len(GROUPED_ORDER)])


def run_text_order(pid, m, n, block_m, block_n, group_m):
    scope = {
        "tl": StubLanguage(pid),
        "M": m,
        "N": n,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "GROUP_M": group_m,
    }
    exec("\n".join(text_order()), scope)
    return scope["pid_m"], scope["pid_n"]


def test_the_package_and_the_kernel_module_import_without_triton_or_torch():
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.stdout.split() == ["False", "False"]


def test_the_text_is_ascii_and_its_digest_is_pinned():
    assert TRITON_SOURCE.isascii() and TRITON_SOURCE.endswith(")\n")
    assert "\t" not in TRITON_SOURCE
    assert source_sha256() == PINNED_SHA256
    assert source_sha256(TRITON_SOURCE) == sha256(TRITON_SOURCE.encode()).hexdigest()
    with pytest.raises(TritonMatmulError, match="ASCII"):
        source_sha256("# é\n")


def test_the_text_holds_two_imports_one_jit_kernel_and_one_launcher():
    body = tree().body

    assert [ast.unparse(node) for node in body[:2]] == [
        "import triton",
        "import triton.language as tl",
    ]
    assert [type(node).__name__ for node in body] == [
        "Import",
        "Import",
        "FunctionDef",
        "FunctionDef",
    ]
    kernel, launcher = function(KERNEL_NAME), function(LAUNCHER_NAME)
    assert [ast.unparse(item) for item in kernel.decorator_list] == ["triton.jit"]
    assert launcher.decorator_list == []
    arguments = [argument.arg for argument in kernel.args.args]
    assert arguments == [
        "a_ptr",
        "b_ptr",
        "c_ptr",
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "stride_cm",
        "stride_cn",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_M",
    ]
    constexpr = {
        argument.arg
        for argument in kernel.args.args
        if argument.annotation is not None
        and ast.unparse(argument.annotation) == "tl.constexpr"
    }
    assert constexpr == {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"}
    assert [argument.arg for argument in launcher.args.kwonlyargs] == list(
        TritonConfig(16, 16, 16, 1, 1, 1).launcher_arguments()
    )


def test_the_kernel_masks_both_loads_and_the_store_and_dots_in_ieee_float32():
    kernel = function(KERNEL_NAME)
    loads = calls(kernel, "tl.load")
    stores = calls(kernel, "tl.store")
    dots = calls(kernel, "tl.dot")

    assert [ast.unparse(call) for call in loads] == [
        "tl.load(a_ptrs, mask=a_mask, other=0.0)",
        "tl.load(b_ptrs, mask=b_mask, other=0.0)",
    ]
    assert [ast.unparse(call) for call in stores] == [
        "tl.store(c_ptrs, acc, mask=c_mask)"
    ]
    assert [ast.unparse(call) for call in dots] == [
        "tl.dot(a, b, acc, input_precision='ieee')"
    ]
    assert calls(kernel, "tl.zeros")[0].keywords[0].value.attr == "float32"
    for line in (*LOAD_MASKS, STORE_MASK, DOT):
        assert f"    {line}\n" in TRITON_SOURCE
    loops = [node for node in ast.walk(kernel) if isinstance(node, ast.For)]
    assert [ast.unparse(loop.iter) for loop in loops] == ["range(0, K, BLOCK_K)"]
    for removed in ("make_block_ptr", "advance", "make_tensor_descriptor", "%"):
        assert f"tl.{removed}" not in TRITON_SOURCE


def test_the_launcher_passes_row_major_strides_and_the_config():
    launch = calls(function(LAUNCHER_NAME), "matmul_kernel[grid]")

    assert len(launch) == 1
    assert [ast.unparse(item) for item in launch[0].args] == [
        "a",
        "b",
        "c",
        "m",
        "n",
        "k",
        "k",
        "1",
        "n",
        "1",
        "n",
        "1",
    ]
    assert {item.arg: ast.unparse(item.value) for item in launch[0].keywords} == {
        "BLOCK_M": "block_m",
        "BLOCK_N": "block_n",
        "BLOCK_K": "block_k",
        "GROUP_M": "group_m",
        "num_warps": "num_warps",
        "num_stages": "num_stages",
    }
    assert (
        "    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)\n"
        in TRITON_SOURCE
    )


def test_the_text_runs_the_grouped_order_of_the_cpu_model():
    assert text_order() == GROUPED_ORDER

    for m, n, block_m, block_n, group_m in (
        (1, 1, 128, 128, 8),
        (1000, 999, 32, 64, 8),
        (255, 257, 64, 32, 8),
        (1000, 999, 128, 32, 3),
        (130, 196, 16, 16, 4),
    ):
        rows, columns = -(-m // block_m), -(-n // block_n)
        for pid in range(rows * columns):
            assert run_text_order(pid, m, n, block_m, block_n, group_m) == (
                program_tile(pid, rows, columns, group_m)
            )


@pytest.mark.parametrize("group_m", range(1, 10))
def test_the_grouped_order_covers_each_tile_once(group_m):
    for rows in range(1, 18):
        for columns in range(1, 18):
            tiles = [
                program_tile(pid, rows, columns, group_m)
                for pid in range(rows * columns)
            ]
            assert sorted(tiles) == [
                (row, column) for row in range(rows) for column in range(columns)
            ]


def test_the_grouped_order_walks_down_a_group_before_it_moves_right():
    order = [program_tile(pid, 10, 3, 4) for pid in range(30)]

    assert order[:8] == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
    assert order[12:16] == [(4, 0), (5, 0), (6, 0), (7, 0)]
    assert order[24:] == [(8, 0), (9, 0), (8, 1), (9, 1), (8, 2), (9, 2)]
    with pytest.raises(TritonMatmulError, match="pid"):
        program_tile(30, 10, 3, 4)
    with pytest.raises(TypeError, match="group_m"):
        program_tile(0, 10, 3, True)


def test_the_declared_configs_are_the_float32_subset_of_the_tutorial():
    assert AUTOTUNE_KEY == ("M", "N", "K")
    assert [config.name for config in AUTOTUNE_CONFIGS] == [
        "m128_n128_k32_g8_w4_s4",
        "m128_n64_k32_g8_w4_s4",
        "m64_n128_k32_g8_w4_s4",
        "m128_n32_k32_g8_w4_s4",
        "m64_n32_k32_g8_w2_s5",
        "m32_n64_k32_g8_w2_s5",
    ]
    assert [config.tile_bytes for config in AUTOTUNE_CONFIGS] == [
        32768,
        24576,
        24576,
        20480,
        12288,
        12288,
    ]
    assert list(CONFIGS_BY_NAME) == [config.name for config in AUTOTUNE_CONFIGS]
    assert AUTOTUNE_CONFIGS[0].meta_parameters() == {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "GROUP_M": 8,
    }
    assert AUTOTUNE_CONFIGS[4].to_mapping() == {
        "name": "m64_n32_k32_g8_w2_s5",
        "block_m": 64,
        "block_n": 32,
        "block_k": 32,
        "group_m": 8,
        "num_warps": 2,
        "num_stages": 5,
        "tile_bytes": 12288,
    }


@pytest.mark.parametrize(
    ("fields", "error", "message"),
    [
        ((48, 64, 32, 8, 4, 4), TritonMatmulError, "power of two"),
        ((8, 64, 32, 8, 4, 4), TritonMatmulError, "block_m"),
        ((64, 512, 32, 8, 4, 4), TritonMatmulError, "block_n"),
        ((64, 64, 32.0, 8, 4, 4), TypeError, "block_k"),
        ((64, 64, 32, 0, 4, 4), TritonMatmulError, "group_m"),
        ((64, 64, 32, 8, 3, 4), TritonMatmulError, "num_warps"),
        ((64, 64, 32, 8, True, 4), TypeError, "num_warps"),
        ((64, 64, 32, 8, 4, 9), TritonMatmulError, "num_stages"),
    ],
)
def test_a_config_outside_the_kernel_rules_is_refused(fields, error, message):
    with pytest.raises(error, match=message):
        TritonConfig(*fields)


def test_the_launch_closed_forms_and_the_shape_bound():
    config = CONFIGS_BY_NAME["m128_n32_k32_g8_w4_s4"]

    assert tile_grid(MatmulShape(1000, 999, 100), config) == (8, 32)
    assert programs(MatmulShape(1000, 999, 100), config) == 256
    assert k_blocks(MatmulShape(1000, 999, 100), config) == 4
    assert (
        programs(MatmulShape(1, 1, 1), config),
        k_blocks(MatmulShape(1, 1, 1), config),
    ) == (1, 1)
    assert k_blocks(MatmulShape(31, 47, 129), config) == 5
    with pytest.raises(TritonMatmulError, match="8192"):
        require_shape(MatmulShape(8193, 1, 1))
    with pytest.raises(TypeError, match="MatmulShape"):
        programs((1, 1, 1), config)
    with pytest.raises(TypeError, match="TritonConfig"):
        k_blocks(MatmulShape(1, 1, 1), config.to_mapping())
