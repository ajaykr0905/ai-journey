import inspect
import re
from hashlib import sha256

from ai_journey.kernel_metal_source import SHADER_SOURCE as DAY_73_SOURCE
from ai_journey.metal_launch_sweep_source import (
    LAYOUT_FIELDS,
    MATMUL,
    SIMD_LAYOUT_PROBE,
    SWEEP_KERNELS,
    SWEEP_SOURCE,
    kernel_bindings,
    matmul_arg_casts,
    matmul_thread,
    source_sha256,
)


def kernel_text(name):
    """Return one kernel's text, from its signature to its closing brace."""

    start = SWEEP_SOURCE.index(f"kernel void {name}(")
    return SWEEP_SOURCE[start : SWEEP_SOURCE.index("\n}\n", start) + 2]


def test_the_matmul_maps_x_to_columns_and_guards_both_axes():
    text = kernel_text(MATMUL)

    for statement in (
        "uint column = position.x;",
        "uint row = position.y;",
        "if (row < m && column < n) {",
        "float sum = 0.0f;",
        "for (uint i = 0; i < k; ++i) {",
        "sum += a[row * k + i] * b[i * n + column];",
        "c[row * n + column] = sum;",
        "uint2 position [[thread_position_in_grid]]",
    ):
        assert statement in text, statement
    for index, name in enumerate(("c", "a", "b")):
        assert f"{name} [[buffer({index})]]" in text
    for index, name in zip((3, 4, 5), "mnk"):
        assert f"constant uint& {name} [[buffer({index})]]" in text
    # Without threadgroup memory or SIMD-group functions, the shape cannot
    # change the arithmetic of a thread.
    assert "threadgroup" not in text and "simd" not in text
    assert matmul_arg_casts() == {3: "int32", 4: "int32", 5: "int32"}


def test_the_simulator_twin_uses_the_same_index_and_guard_statements():
    twin = inspect.getsource(matmul_thread)

    for statement in (
        "column = t.block_idx.x * t.block_dim.x + t.thread_idx.x",
        "row = t.block_idx.y * t.block_dim.y + t.thread_idx.y",
        "if row < m and column < n:",
        "a[row * k + i] * b[i * n + column]",
        "c[row * n + column] = total",
    ):
        assert statement in twin, statement


def test_the_probe_records_five_words_per_thread_at_its_linear_index():
    text = kernel_text(SIMD_LAYOUT_PROBE)

    assert LAYOUT_FIELDS == ("simd_group", "lane", "x", "y", "simd_width")
    stores = re.findall(r"out\[(5 \* linear(?: \+ \d)?)\] = ([a-z_.]+);", text)
    assert stores == [
        ("5 * linear", "simd_group"),
        ("5 * linear + 1", "lane"),
        ("5 * linear + 2", "position.x"),
        ("5 * linear + 3", "position.y"),
        ("5 * linear + 4", "width"),
    ]
    for attribute in (
        "thread_index_in_threadgroup",
        "thread_position_in_threadgroup",
        "simdgroup_index_in_threadgroup",
        "thread_index_in_simdgroup",
        "threads_per_simdgroup",
    ):
        assert f"[[{attribute}]]" in text, attribute


def test_bindings_and_digest_describe_the_exact_source():
    assert SWEEP_KERNELS == (MATMUL, SIMD_LAYOUT_PROBE)
    assert [item["name"] for item in kernel_bindings()] == list(SWEEP_KERNELS)
    assert kernel_bindings()[0]["buffers"] == ["c", "a", "b", "m", "n", "k"]
    assert kernel_bindings()[0]["scalar_buffers"] == [3, 4, 5]
    assert source_sha256() == sha256(SWEEP_SOURCE.encode("ascii")).hexdigest()
    assert SWEEP_SOURCE.isascii()
    # The 1D sweep runs the Day 73 vector_add; this source does not copy it.
    assert "kernel void vector_add(" in DAY_73_SOURCE
    assert "vector_add" not in SWEEP_SOURCE
