import re

import numpy
import pytest

from ai_journey.gpu_scan_source import (
    ADD_OFFSETS,
    ADD_SOURCE,
    BLOCK,
    KERNEL_BODY,
    KERNELS,
    MAX_LENGTH,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    ScanKernel,
    add_mapping,
    barrier_heads,
    block_count,
    dispatches,
    guard_elements,
    level_lengths,
    library_sha256,
    library_source,
    require_length,
    rounding_depth,
    uniform_barriers,
)
from ai_journey.matmul_tiled_source import BARRIER, HEADER
from ai_journey.matmul_tiled_variants import text_sha256
from test_kernel_metal_runtime import requires_metal


def test_each_kernel_scans_one_block_of_1024_elements_in_its_own_group():
    rows = {
        name: (
            kernel.threads_per_group,
            kernel.elements_per_thread,
            kernel.threadgroup_bytes,
            kernel.simd_width,
            kernel.barriers_per_block,
            kernel.block_depth,
            kernel.replay,
        )
        for name, kernel in KERNELS.items()
    }

    assert rows == {
        S0_HILLIS_STEELE: (1024, 1, 8192, None, 11, 10, True),
        S1_BLELLOCH: (512, 2, 4096, None, 22, 19, True),
        S2_SIMD: (1024, 1, 128, 32, 2, 62, False),
    }
    for kernel in KERNELS.values():
        assert kernel.block == BLOCK
        assert kernel.arg_casts == {3: "int32"}
        assert kernel.source_sha256 == text_sha256(kernel.source)
        assert kernel.to_mapping()["source_sha256"] == kernel.source_sha256
        assert "kernel.source" not in kernel.to_mapping()


def test_the_library_is_the_day_76_header_then_every_kernel_in_order():
    source = library_source()
    names = re.findall(r"kernel void (\w+)\(", source)

    assert source.startswith(HEADER)
    assert names == [*KERNELS, ADD_OFFSETS]
    assert source == HEADER + "".join(
        "\n" + text for text in (*(k.source for k in KERNELS.values()), ADD_SOURCE)
    )
    assert library_sha256() == text_sha256(source)
    assert add_mapping() == {
        "name": ADD_OFFSETS,
        "group_size": 256,
        "threadgroup_bytes": 0,
        "source_sha256": text_sha256(ADD_SOURCE),
    }
    assert source.isascii() and "\t" not in source


def test_the_simd_kernel_uses_only_the_named_simd_group_functions():
    text = KERNELS[S2_SIMD].source

    assert text.count("simd_prefix_inclusive_sum(") == 2
    assert text.count("simd_shuffle_up(") == 1
    assert "simd_prefix_exclusive_sum" not in text
    for kernel in (S0_HILLIS_STEELE, S1_BLELLOCH):
        assert "simd_" not in KERNELS[kernel].source


def test_every_barrier_sits_in_the_body_or_a_loop_with_uniform_bounds():
    for kernel in KERNELS.values():
        heads = barrier_heads(kernel.source)
        assert heads and all(stack[0] == KERNEL_BODY for stack in heads)
        assert all(head.startswith("for (") for stack in heads for head in stack[1:])
        assert "return" not in kernel.source
        # Each guard on the thread holds loads, stores or adds, never a barrier.
        for stack in heads:
            assert not any(head.startswith(("if", "} else")) for head in stack)
    assert barrier_heads(ADD_SOURCE) == ()


def _in_step(*lines):
    """Insert lines into the S0 step loop, after its buffer swap and before its barrier."""

    added = "".join(f"        {line}\n" for line in lines)
    return lambda text: text.replace(
        "        source ^= 1u;\n", f"        source ^= 1u;\n{added}"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        # Each of the next nine texts compiles with Metal and lets a thread skip
        # a barrier that the other threads of its group wait at.
        (_in_step("if (t < offset) {", "    break;", "}"), "break, continue or goto"),
        (
            _in_step("if (t < offset) {", "    continue;", "}"),
            "break, continue or goto",
        ),
        (
            _in_step("if (t == 0u) {", "    offset = 1024u;", "}"),
            "write or alias its loop variable 'offset'",
        ),
        (
            _in_step(
                "thread uint* step = &offset;",
                "if (t == 0u) {",
                "    *step = 1024u;",
                "}",
            ),
            "write or alias its loop variable 'offset'",
        ),
        (
            lambda text: text.replace(
                "    for (uint offset", "    if (t < 512u)\n    for (uint offset"
            ),
            "block must open after a complete statement or an opening brace",
        ),
        (
            lambda text: text.replace(
                f"        {BARRIER}\n    }}\n",
                f"        if (t < 512u)\n            {BARRIER}\n    }}\n",
            ),
            "alone, after a complete statement or an opening brace",
        ),
        (
            lambda text: text.replace(
                f"        {BARRIER}\n    }}\n",
                f"        if (t < 512u) {BARRIER}\n    }}\n",
            ),
            "alone, after a complete statement or an opening brace",
        ),
        (
            _in_step(
                "if (t < 512u) {",
                "    threadgroup_barrier(mem_flags::mem_none);",
                "}",
            ),
            "alone, after a complete statement or an opening brace",
        ),
        (
            lambda text: f"void wait_here() {{\n    {BARRIER}\n}}\n"
            + _in_step("if (t < 512u) {", "    wait_here();", "}")(text),
            "must define one function",
        ),
        (
            lambda text: text.replace(
                "        if (t >= offset) {\n",
                f"        if (t >= offset) {{\n            {BARRIER}\n",
            ),
            "not a loop with the same bounds",
        ),
        (
            lambda text: text.replace(
                "for (uint offset = 1u; offset < 1024u; offset <<= 1u) {",
                "for (uint offset = 1u; offset < t + 1u; offset <<= 1u) {",
            ),
            "not a loop with the same bounds",
        ),
        (
            lambda text: text.replace(
                "    uint source = 0u;\n",
                "    if (i >= n) {\n        return;\n    }\n    uint source = 0u;\n",
            ),
            "must not return early",
        ),
        (lambda text: text + "}\n", "closes a block it never opened"),
        (lambda text: text.rstrip("}\n"), "leaves a block open"),
    ],
)
def test_the_barrier_check_refuses_a_barrier_that_some_threads_skip(change, message):
    text = change(KERNELS[S0_HILLIS_STEELE].source)

    with pytest.raises(GpuScanError, match=message):
        uniform_barriers(text)
    with pytest.raises(GpuScanError, match=message):
        ScanKernel(
            S0_HILLIS_STEELE, "hillis_steele", 1024, 1, 8192, None, 10, True, text
        )


def test_the_flow_rules_bind_only_a_kernel_that_waits_at_a_barrier():
    early = "kernel void scan_x(uint i [[thread_position_in_grid]]) {\n"
    early += "    if (i > 7u) {\n        return;\n    }\n}\n"

    assert uniform_barriers(early) == 0
    assert uniform_barriers(ADD_SOURCE) == 0


def _hillis_steele_depth() -> numpy.ndarray:
    """The adds on the deepest path to each output, by a max-plus model."""

    depth = numpy.zeros(BLOCK, dtype=numpy.int64)
    offset = 1
    while offset < BLOCK:
        later = depth.copy()
        later[offset:] = numpy.maximum(depth[offset:], depth[:-offset]) + 1
        depth, offset = later, offset * 2
    return depth


def _blelloch_depth() -> numpy.ndarray:
    tree = numpy.zeros(BLOCK, dtype=numpy.int64)
    stride = 1
    while stride < BLOCK:
        right = numpy.arange(2 * stride - 1, BLOCK, 2 * stride)
        tree[right] = numpy.maximum(tree[right - stride], tree[right]) + 1
        stride *= 2
    total = int(tree[-1])
    tree[-1] = 0
    stride = BLOCK // 2
    while stride:
        right = numpy.arange(2 * stride - 1, BLOCK, 2 * stride)
        carry = tree[right - stride].copy()
        tree[right - stride] = tree[right]
        tree[right] = numpy.maximum(carry, tree[right]) + 1
        stride //= 2
    return numpy.append(tree[1:], total)


def _serial_simd_depth() -> numpy.ndarray:
    """The worst order of a SIMD-group prefix sum: one lane after another."""

    lanes = numpy.arange(32)
    running = 31 + lanes  # a serial scan of the totals, each of depth 31
    earlier = numpy.concatenate([[0], running[:-1]])
    groups = numpy.repeat(numpy.arange(32), 32)
    local = numpy.tile(lanes, 32)
    return numpy.maximum(local, earlier[groups]) + 1


def test_each_declared_block_depth_covers_its_order():
    assert int(_hillis_steele_depth().max()) == KERNELS[S0_HILLIS_STEELE].block_depth
    assert int(_blelloch_depth().max()) == KERNELS[S1_BLELLOCH].block_depth
    assert int(_serial_simd_depth().max()) == KERNELS[S2_SIMD].block_depth


def _multi_depth(length: int, block: numpy.ndarray) -> numpy.ndarray:
    """Compose a block model over levels: each later element adds its offset."""

    count = block_count(length)
    local = numpy.tile(block, count)[:length]
    if count == 1:
        return local
    # Every block total enters the next level with the depth of the last slot.
    offsets = _multi_depth(count, block) + int(block[BLOCK - 1])
    out = local.copy()
    index = numpy.arange(length) // BLOCK
    later = index > 0
    out[later] = numpy.maximum(local[later], offsets[index[later] - 1]) + 1
    return out


@pytest.mark.parametrize("length", [1, 1023, 1024, 1025, 4097, 1 << 20, 1048577])
def test_the_multi_level_depth_covers_every_output(length):
    for name, block in (
        (S0_HILLIS_STEELE, _hillis_steele_depth()),
        (S1_BLELLOCH, _blelloch_depth()),
        (S2_SIMD, _serial_simd_depth()),
    ):
        assert int(_multi_depth(length, block).max()) <= rounding_depth(name, length)
    levels = len(level_lengths(length))
    assert rounding_depth(S0_HILLIS_STEELE, length) == 10 * levels + levels - 1


def test_lengths_split_into_levels_dispatches_and_guard_tails():
    rows = {
        length: (level_lengths(length), dispatches(length), guard_elements(length))
        for length in (1, 1000, 1024, 1025, 1 << 16, 1 << 20, 1048577, 5000003)
    }

    assert rows == {
        1: ((1,), 1, 2047),
        1000: ((1000,), 1, 1048),
        1024: ((1024,), 1, 1024),
        1025: ((1025, 2), 3, 2047),
        1 << 16: ((65536, 64), 3, 1024),
        1 << 20: ((1048576, 1024), 3, 1024),
        1048577: ((1048577, 1025, 2), 5, 2047),
        5000003: ((5000003, 4883, 5), 5, 1213),
    }
    assert level_lengths(MAX_LENGTH) == (1 << 24, 1 << 14, 16)
    for value, error in ((0, GpuScanError), (MAX_LENGTH + 1, GpuScanError)):
        with pytest.raises(error):
            require_length(value)
    for value in (True, 2.0, "8"):
        with pytest.raises(TypeError):
            require_length(value)
    with pytest.raises(GpuScanError, match="not a registered"):
        rounding_depth("scan_s9_other", 8)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "matmul_scan", "scan_ and a short slug"),
        ("threads_per_group", 256, "exactly 1024 elements"),
        ("simd_width", 64, "needs 32 lanes"),
        ("source", "kernel void other(", "needs ASCII source that defines it"),
        ("block_depth", 9, "block_depth"),
    ],
)
def test_a_kernel_with_a_wrong_field_is_refused(field, value, message):
    fields = dict(KERNELS[S2_SIMD].__dict__)
    fields[field] = value

    with pytest.raises(GpuScanError, match=message):
        ScanKernel(**fields)


@requires_metal
def test_this_gpu_compiles_the_library_with_the_declared_memory():
    import torch

    library = torch.mps.compile_shader(library_source())

    for name, kernel in KERNELS.items():
        pipeline = getattr(library, name)
        assert pipeline.static_thread_group_memory_length == kernel.threadgroup_bytes
        assert pipeline.max_threads_per_threadgroup >= kernel.threads_per_group
        assert pipeline.thread_execution_width == 32
    assert getattr(library, ADD_OFFSETS).static_thread_group_memory_length == 0
