"""Tests of the pinned Day 81 Triton text and its declared configs, without triton."""

import ast
import sys

import pytest

from ai_journey import triton_flash_kernel
from ai_journey.triton_flash_kernel import (
    CAUSAL_MASK,
    CONFIGS_BY_NAME,
    FLASH_CONFIGS,
    KERNEL_NAME,
    LAUNCHER_NAME,
    LOAD_MASKS,
    ONLINE_SOFTMAX,
    SCORE,
    STORES,
    TRITON_SOURCE,
    FlashConfig,
    TritonFlashError,
    float32_scale,
    key_blocks,
    padded_rows,
    query_blocks,
    source_sha256,
)


def functions():
    tree = ast.parse(TRITON_SOURCE)
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_the_text_parses_into_one_jit_kernel_and_one_launcher():
    found = functions()

    assert set(found) == {KERNEL_NAME, LAUNCHER_NAME}
    kernel = found[KERNEL_NAME]
    assert [ast.unparse(item) for item in kernel.decorator_list] == ["triton.jit"]
    constexprs = [
        arg.arg
        for arg in kernel.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation) == "tl.constexpr"
    ]
    assert constexprs == ["HEAD_DIM", "BLOCK_M", "BLOCK_N", "CAUSAL"]
    assert TRITON_SOURCE.isascii() and TRITON_SOURCE.endswith("\n")
    assert len(source_sha256()) == 64


def test_the_text_holds_the_online_softmax_inside_the_key_loop():
    lines = TRITON_SOURCE.splitlines()
    loop = lines.index("    for start_n in range(0, N, BLOCK_N):")
    after = lines.index("    acc = acc / l_i[:, None]")

    body = [line.strip() for line in lines[loop + 1 : after]]
    assert body[-len(ONLINE_SOFTMAX) :] == list(ONLINE_SOFTMAX)
    assert SCORE in body and CAUSAL_MASK in body
    assert 's = tl.where(keep, s, float("-inf"))' in body
    assert all(mask in TRITON_SOURCE for mask in LOAD_MASKS)
    assert all(store in TRITON_SOURCE for store in STORES)
    # No score or probability is ever stored: the only stores are O, m and l.
    assert TRITON_SOURCE.count("tl.store(") == 3
    assert TRITON_SOURCE.count('input_precision="ieee"') == 2


def test_the_launcher_grid_is_query_blocks_by_heads():
    launcher = ast.unparse(functions()[LAUNCHER_NAME])

    assert "grid = (triton.cdiv(n, block_m), bh)" in launcher
    assert "n * head_dim, head_dim, rows * head_dim, head_dim, rows" in launcher


def test_the_module_never_imports_triton():
    assert "triton" not in triton_flash_kernel.__dict__
    tree = ast.parse(open(triton_flash_kernel.__file__, encoding="utf-8").read())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "triton" in ast.unparse(node).split("import")[-1].split()
    ]
    assert imports == []
    assert "triton" not in sys.modules or sys.modules["triton"] is not None


def test_the_declared_configs_and_their_geometry():
    assert [config.name for config in FLASH_CONFIGS] == [
        "m128_n64_w8_s3",
        "m64_n64_w4_s3",
        "m64_n32_w4_s4",
        "m16_n16_w1_s1",
    ]
    assert CONFIGS_BY_NAME["m64_n32_w4_s4"] is FLASH_CONFIGS[2]
    config = FLASH_CONFIGS[0]
    # Q tile 128 x 64, K and V tiles 64 x 64, score tile 128 x 64, float32.
    assert config.tile_bytes(64) == (128 * 64 + 2 * 64 * 64 + 128 * 64) * 4
    assert config.to_mapping()["tile_bytes"]["128"] == config.tile_bytes(128)
    assert config.meta_parameters() == {"BLOCK_M": 128, "BLOCK_N": 64}
    assert (query_blocks(129, config), key_blocks(129, config)) == (2, 3)
    assert padded_rows(129, config) == 256
    assert padded_rows(1, FLASH_CONFIGS[3]) == 16


@pytest.mark.parametrize(
    ("fields", "error"),
    (
        ((8, 16, 4, 3), TritonFlashError),
        ((256, 16, 4, 3), TritonFlashError),
        ((48, 16, 4, 3), TritonFlashError),
        ((64, 64, 3, 3), TritonFlashError),
        ((64, 64, 4, 0), TritonFlashError),
        ((64, 64, 4.0, 3), TypeError),
        ((True, 64, 4, 3), TypeError),
    ),
)
def test_a_config_refuses_bad_fields(fields, error):
    with pytest.raises(error):
        FlashConfig(*fields)


def test_the_scale_takes_only_a_declared_head_dim():
    assert float32_scale(16) == 0.25
    with pytest.raises(TritonFlashError, match="head_dim"):
        float32_scale(48)
    with pytest.raises(TypeError):
        float32_scale(64.0)
