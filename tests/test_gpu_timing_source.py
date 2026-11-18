import re
from hashlib import sha256

from ai_journey.gpu_timing_source import (
    EMPTY_KERNEL,
    EMPTY_SOURCE,
    NEVER_POSITION,
    empty_kernel_mapping,
    empty_launch,
    empty_source_sha256,
)
from ai_journey.matmul_metal_source import SHADER_SOURCE, source_sha256
from test_kernel_metal_runtime import requires_metal


def test_the_source_defines_one_kernel_with_one_buffer():
    assert EMPTY_SOURCE.isascii()
    assert re.findall(r"kernel void (\w+)\(", EMPTY_SOURCE) == [EMPTY_KERNEL]
    assert EMPTY_SOURCE.count("[[buffer(") == 1
    assert "device uint* sink [[buffer(0)]]" in EMPTY_SOURCE
    assert "uint position [[thread_position_in_grid]]" in EMPTY_SOURCE


def test_the_store_depends_on_a_position_that_no_launch_reaches():
    launch = empty_launch()

    assert f"if (position == 0x{NEVER_POSITION:X}u)" in EMPTY_SOURCE
    assert "sink[0] = position;" in EMPTY_SOURCE
    assert launch.launched_threads == launch.active_threads == 1
    assert launch.launch_arguments() == {"threads": 1, "group_size": 1}
    assert NEVER_POSITION >= launch.launched_threads


def test_the_digest_covers_the_exact_text_and_leaves_day_74_alone():
    assert empty_source_sha256() == sha256(EMPTY_SOURCE.encode("ascii")).hexdigest()
    assert EMPTY_KERNEL not in SHADER_SOURCE
    assert empty_source_sha256() != source_sha256()


def test_the_mapping_records_the_bindings_and_the_launch():
    mapping = empty_kernel_mapping()

    assert mapping["name"] == EMPTY_KERNEL
    assert mapping["buffers"] == ["sink"]
    assert mapping["never_position"] == 0xFFFFFFFF
    assert mapping["launch"]["threads"] == [1]


@requires_metal
def test_the_empty_kernel_compiles_and_leaves_its_buffer_unchanged():
    import torch

    library = torch.mps.compile_shader(EMPTY_SOURCE)
    sink = torch.full((4,), 7, dtype=torch.int32, device="mps")

    getattr(library, EMPTY_KERNEL)(sink, **empty_launch().launch_arguments())
    torch.mps.synchronize()

    assert sink.cpu().tolist() == [7, 7, 7, 7]
    assert getattr(library, EMPTY_KERNEL).max_threads_per_threadgroup >= 1
