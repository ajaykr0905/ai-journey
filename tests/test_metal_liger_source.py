import dataclasses
import re

import pytest

from ai_journey.matmul_tiled_variants import text_sha256
from ai_journey.metal_liger_source import (
    CROSS_ENTROPY,
    CROSS_ENTROPY_KERNEL,
    FAMILIES,
    KERNELS,
    MAX_REGISTER_ROW,
    MAX_STREAM_ROW,
    PROBES,
    RMSNORM,
    RMSNORM_KERNEL,
    SWIGLU,
    SWIGLU_KERNEL,
    MetalLigerError,
    columns_per_thread,
    family_kernel,
    launch_group,
    liger_library,
    library_sha256,
)


def test_every_family_has_one_fused_kernel_and_every_probe_one_kernel():
    assert [family_kernel(family).name for family in FAMILIES] == [
        RMSNORM_KERNEL,
        SWIGLU_KERNEL,
        CROSS_ENTROPY_KERNEL,
    ]
    assert set(KERNELS) == {RMSNORM_KERNEL, SWIGLU_KERNEL, CROSS_ENTROPY_KERNEL} | set(
        PROBES.values()
    )
    for name, kernel in KERNELS.items():
        assert kernel.source_sha256 == text_sha256(kernel.source)
        assert kernel.source in liger_library()
        assert re.findall(r"kernel void (\w+)\(", kernel.source) == [name]
    assert library_sha256() == text_sha256(liger_library())
    assert liger_library().startswith("#include <metal_stdlib>\n")


def test_the_kernels_keep_their_declared_barriers_and_threadgroup_bytes():
    assert KERNELS[RMSNORM_KERNEL].barriers == 1
    assert KERNELS[RMSNORM_KERNEL].threadgroup_bytes == 32 * 4
    assert KERNELS[SWIGLU_KERNEL].barriers == 0
    assert KERNELS[SWIGLU_KERNEL].threadgroup_bytes == 0
    assert KERNELS[CROSS_ENTROPY_KERNEL].barriers == 1
    assert KERNELS[CROSS_ENTROPY_KERNEL].threadgroup_bytes == 2 * 32 * 4
    text = KERNELS[CROSS_ENTROPY_KERNEL].source
    assert "float margin = row_max - x[start + targets[row]];" in text
    assert "loss[row] = margin + precise::log(total);" in text
    assert "y[" not in text
    assert "precise::rsqrt(mean + 1.0e-6f)" in KERNELS[RMSNORM_KERNEL].source


def test_a_barrier_inside_a_thread_condition_is_refused():
    kernel = KERNELS[RMSNORM_KERNEL]
    bad = kernel.source.replace(
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "    if (lane == 0u) {\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n",
    )

    with pytest.raises(MetalLigerError, match="barrier sits inside"):
        dataclasses.replace(kernel, source=bad)
    with pytest.raises(MetalLigerError, match="wrong barrier count"):
        dataclasses.replace(kernel, barriers=2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "softmax_x"}, "liger_"),
        ({"family": SWIGLU}, "its own family"),
        ({"buffers": ()}, "name its buffers"),
        ({"threadgroup_bytes": -1}, "threadgroup_bytes"),
        ({"summary": "café"}, "ASCII"),
        ({"source": "// nothing\n"}, "does not define"),
    ],
)
def test_a_kernel_refuses_an_invalid_field(changes, message):
    with pytest.raises(MetalLigerError, match=message):
        dataclasses.replace(KERNELS[RMSNORM_KERNEL], **changes)


def test_the_launch_rule_fills_16_slots_and_stops_at_1024_threads():
    assert launch_group(RMSNORM, 1) == 32
    assert launch_group(RMSNORM, 512) == 32
    assert launch_group(RMSNORM, 513) == 64
    assert launch_group(RMSNORM, 4096) == 256
    assert launch_group(RMSNORM, MAX_REGISTER_ROW) == 1024
    assert launch_group(SWIGLU, 2049) == 256
    assert launch_group(CROSS_ENTROPY, 32000) == 1024
    assert launch_group(CROSS_ENTROPY, 50257) == 1024
    assert columns_per_thread(50257, 1024) == 50
    assert columns_per_thread(1, 32) == 1
    with pytest.raises(MetalLigerError, match="longest row"):
        launch_group(RMSNORM, MAX_REGISTER_ROW + 1)
    with pytest.raises(MetalLigerError, match="longest row"):
        launch_group(CROSS_ENTROPY, MAX_STREAM_ROW + 1)
    with pytest.raises(MetalLigerError, match="kernel family"):
        launch_group("layernorm", 8)
    with pytest.raises(TypeError):
        launch_group(SWIGLU, True)
