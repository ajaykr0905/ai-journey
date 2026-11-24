import dataclasses
import re

import pytest

from ai_journey.matmul_tiled_source import BARRIER
from ai_journey.matmul_tiled_variants import text_sha256
from ai_journey.metal_flash_source import (
    CAUSAL,
    F1,
    FLASH_GROUP,
    FULL,
    KERNELS,
    MODES,
    N0,
    PROBE_EXP,
    TILE_LOOP,
    MetalFlashError,
    flash_library,
    kernel_names,
    library_sha256,
    require_uniform_barriers,
)
from ai_journey.metal_fused_source import MetalFusedError


def test_every_side_and_mode_has_its_kernels_and_one_probe():
    names = {name for mode in MODES for name in kernel_names(N0, mode)}
    names |= {name for mode in MODES for name in kernel_names(F1, mode)}

    assert set(KERNELS) == names | {PROBE_EXP}
    assert kernel_names(N0, CAUSAL) == (
        "flash_n0_scores_causal",
        "flash_n0_softmax_causal",
        "flash_n0_pv_causal",
    )
    assert kernel_names(F1, FULL) == ("flash_f1",)
    for name, kernel in KERNELS.items():
        assert kernel.source_sha256 == text_sha256(kernel.source)
        assert kernel.source in flash_library()
        assert re.findall(r"kernel void (\w+)\(", kernel.source) == [name]
    assert library_sha256() == text_sha256(flash_library())
    assert flash_library().startswith("#include <metal_stdlib>\n")
    assert issubclass(MetalFlashError, MetalFusedError)


def test_the_kernels_keep_their_declared_groups_barriers_and_memory():
    f1 = KERNELS["flash_f1"]
    assert (f1.group_size, f1.threadgroup_bytes, f1.barriers) == (128, 16384, 2)
    assert FLASH_GROUP == 128
    softmax = KERNELS["flash_n0_softmax"]
    assert (softmax.group_size, softmax.threadgroup_bytes, softmax.barriers) == (
        256,
        64,
        2,
    )
    pv = KERNELS["flash_n0_pv"]
    assert (pv.group_size, pv.threadgroup_bytes, pv.barriers) == (64, 0, 0)
    assert KERNELS["flash_n0_scores"].barriers == 1


def test_the_flash_kernel_rescales_online_and_never_writes_scores():
    full, causal = KERNELS["flash_f1"].source, KERNELS["flash_f1_causal"].source

    assert TILE_LOOP in full
    assert "float alpha = precise::exp(m - next);" in full
    assert "l *= alpha;" in full and "acc[t] *= alpha;" in full
    assert "precise::divide(acc[t], l)" in full
    assert "s[" in full and "device float* s" not in full
    assert "uint end = length;" in full
    assert "uint end = min(length, first + 32u);" in causal
    assert "key < length && key <= row" in causal
    assert "if (key < length) {" in full


def test_the_naive_kernels_write_the_scores_to_device_memory():
    assert "device float* s [[buffer(0)]]" in KERNELS["flash_n0_scores"].source
    assert "uint limit = row + 1u;" in KERNELS["flash_n0_softmax_causal"].source
    assert "uint limit = length;" in KERNELS["flash_n0_pv"].source


def test_a_barrier_inside_a_thread_condition_is_refused():
    kernel = KERNELS["flash_f1"]
    text = kernel.source.replace(
        f"        {BARRIER}\n        float s[32];",
        f"        if (live) {{\n            {BARRIER}\n        }}\n        float s[32];",
    )
    assert text != kernel.source

    with pytest.raises(MetalFlashError, match="depends on the thread"):
        require_uniform_barriers(text)
    with pytest.raises(MetalFlashError, match="depends on the thread"):
        dataclasses.replace(kernel, source=text)


def test_a_barrier_in_another_loop_or_an_open_block_is_refused():
    loop = "for (uint j = lane; j < limit; j += size) {"
    with pytest.raises(MetalFlashError, match="depends on the thread"):
        require_uniform_barriers(f"kernel void k() {{\n{loop}\n{BARRIER}\n}}\n}}\n")
    with pytest.raises(MetalFlashError, match="not closed"):
        require_uniform_barriers("kernel void k() {\n")
    with pytest.raises(MetalFlashError, match="no open block"):
        require_uniform_barriers("}\n")
    assert require_uniform_barriers(f"kernel void k() {{\n{BARRIER}\n}}\n") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"name": "softmax_f1"},
        {"variant": N0},
        {"mode": CAUSAL},
        {"barriers": 1},
        {"buffers": ()},
        {"group_size": 2048},
        {"summary": "café"},
    ],
)
def test_a_kernel_record_that_disagrees_with_its_text_is_refused(change):
    with pytest.raises(MetalFlashError):
        dataclasses.replace(KERNELS["flash_f1"], **change)


def test_an_unknown_variant_or_mode_has_no_kernels():
    with pytest.raises(MetalFlashError, match="variant"):
        kernel_names("f2", FULL)
    with pytest.raises(MetalFlashError, match="mode"):
        kernel_names(F1, "sliding")
