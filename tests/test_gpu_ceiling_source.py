import re
from hashlib import sha256

import pytest

from ai_journey.gpu_ceiling_source import (
    BANK,
    BANK_COLUMNS,
    BANK_LANES,
    BANK_LOADS_PER_TRIP,
    BANK_MAX_STRIDE,
    BANK_ROW_WORDS,
    BANK_SHIFT_PERIOD,
    BANK_TILE_ROWS,
    BANDWIDTH,
    FMA,
    FMA_CHAIN_COUNTS,
    FMA_STEPS_PER_TRIP,
    KERNELS,
    SHADER_SOURCE,
    STREAM_KERNELS,
    GpuCeilingError,
    bank_kernel_name,
    bank_tile_bytes,
    fma_kernel_name,
    require_count,
    source_sha256,
)
from ai_journey.kernel_metal_geometry import KernelMetalError
from test_kernel_metal_runtime import requires_metal


def kernel_text(name):
    """Return one kernel's text, from its comment to the next kernel."""

    blocks = re.split(r"\n(?=\n// )", SHADER_SOURCE)
    (block,) = [text for text in blocks if f"kernel void {name}(" in text]
    return block.strip()


def loop_body(text, header):
    start = text.index(header) + len(header)
    depth, index = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        index += 1
    return text[start : index - 1]


def test_each_fma_loop_only_steps_its_chains_in_registers():
    assert SHADER_SOURCE.count("#define FMA_STEP(x) x = fma(x, a, b);") == 1
    macro = re.search(r"#define EIGHT_TIMES\(steps\) (.*)\n", SHADER_SOURCE)
    assert macro.group(1).split() == ["steps"] * FMA_STEPS_PER_TRIP

    for chains in FMA_CHAIN_COUNTS:
        body = loop_body(
            kernel_text(fma_kernel_name(chains)),
            "for (uint t = 0u; t < trips; ++t) {",
        )
        steps = re.findall(r"FMA_STEP\((x\d+)\)", body)
        assert steps == [f"x{j}" for j in range(chains)], chains
        # No buffer, index or address appears inside the timed loop.
        assert not re.search(r"\[|start|out|coefficients|device", body), chains


def test_every_chain_feeds_the_one_output_of_its_thread():
    for chains in FMA_CHAIN_COUNTS:
        text = kernel_text(fma_kernel_name(chains))
        read = re.findall(r"as_type<uint>\((x\d+)\)", text)
        assert read == [f"x{j}" for j in range(chains)], chains
        assert text.count("out[i] = bits;") == 1
        assert text.count("if (i < n) {") == 1


def test_both_bank_kernels_run_the_same_instructions_but_the_row_length():
    def shape(row_words):
        text = kernel_text(bank_kernel_name(row_words))
        return re.sub(r"\d+", "#", text)

    assert shape(32) == shape(33)
    for row_words in BANK_ROW_WORDS:
        text = kernel_text(bank_kernel_name(row_words))
        assert f"threadgroup uint tile[{BANK_TILE_ROWS * row_words}];" in text
        loads = re.findall(r"tile\[q \+ (\d+)u\]", text)
        assert [int(offset) for offset in loads] == [
            load * row_words for load in range(BANK_LOADS_PER_TRIP)
        ]
        assert f"(k % {BANK_SHIFT_PERIOD}u) * {row_words}u" in text
        assert bank_tile_bytes(row_words) == 4 * BANK_TILE_ROWS * row_words


def test_every_bank_read_stays_inside_the_tile():
    rows_down = BANK_SHIFT_PERIOD - 1 + BANK_LOADS_PER_TRIP - 1
    for row_words in BANK_ROW_WORDS:
        deepest = 0
        for stride in range(1, BANK_MAX_STRIDE + 1):
            for lane in range(BANK_LANES):
                logical = lane * stride
                first = logical // BANK_COLUMNS * row_words + logical % BANK_COLUMNS
                deepest = max(deepest, first + rows_down * row_words)
        # Lane 31 at stride 32 reads row 31 + 78, the last of the tile's rows
        # that a read can reach.
        assert deepest == (BANK_TILE_ROWS - 3) * row_words


def test_kernel_specs_name_their_bindings_and_pass_32_bit_scalars():
    names = [fma_kernel_name(chains) for chains in FMA_CHAIN_COUNTS]
    names += list(STREAM_KERNELS)
    names += [bank_kernel_name(row_words) for row_words in BANK_ROW_WORDS]
    assert list(KERNELS) == names
    for name, spec in KERNELS.items():
        assert f"kernel void {name}(" in SHADER_SOURCE
        assert spec.arg_casts == {index: "int32" for index in spec.scalar_buffers}
        for index in spec.scalar_buffers:
            binding = f"constant uint& {spec.buffers[index]} [[buffer({index})]]"
            assert binding in kernel_text(name), (name, binding)
    assert {spec.family for spec in KERNELS.values()} == {FMA, BANDWIDTH, BANK}
    assert KERNELS["stream_copy"].to_mapping() == {
        "name": "stream_copy",
        "family": "bandwidth",
        "buffers": ["out", "in", "n", "threads"],
        "scalar_buffers": [2, 3],
    }


@pytest.mark.parametrize("value", [0, 3, 64, True, 8.0, "8"])
def test_unknown_chain_counts_and_row_lengths_are_refused(value):
    with pytest.raises(GpuCeilingError, match="chains must be one of"):
        fma_kernel_name(value)
    with pytest.raises(GpuCeilingError, match="row_words must be 32 or 33"):
        bank_kernel_name(value)
    assert issubclass(GpuCeilingError, KernelMetalError)


def test_counts_reuse_the_metal_bound_rule_with_this_labs_error():
    assert require_count(32, "stride", 1, 32) == 32
    with pytest.raises(GpuCeilingError, match=r"stride must be in \[1, 32\]"):
        require_count(33, "stride", 1, 32)
    for value in (True, 2.0):
        with pytest.raises(TypeError, match="stride must be an integer"):
            require_count(value, "stride", 1, 32)


def test_the_source_is_ascii_and_its_digest_covers_the_exact_text():
    assert source_sha256() == sha256(SHADER_SOURCE.encode("ascii")).hexdigest()
    assert SHADER_SOURCE.startswith("#include <metal_stdlib>\n")


@requires_metal
def test_the_source_compiles_and_each_pipeline_reports_its_limits():
    import torch

    library = torch.mps.compile_shader(SHADER_SOURCE)

    for name, spec in KERNELS.items():
        kernel = getattr(library, name)
        assert kernel.max_threads_per_threadgroup >= 256, name
        assert kernel.thread_execution_width == 32, name
        memory = kernel.static_thread_group_memory_length
        if spec.family == BANK:
            assert memory == bank_tile_bytes(int(name.rsplit("_", 1)[1])), name
        else:
            assert memory == 0, name
