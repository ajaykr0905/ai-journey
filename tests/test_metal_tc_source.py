import dataclasses
import re

import pytest

from ai_journey.matmul_tiled_source import BARRIER
from ai_journey.matmul_tiled_variants import text_sha256
from ai_journey.metal_flash_source import MetalFlashError
from ai_journey.metal_tc_source import (
    A_SLOTS,
    B_SLOTS,
    BFLOAT_LIBRARY,
    GROUP,
    KERNEL_NAME,
    KERNELS,
    LIBRARIES,
    M1,
    M2,
    M3,
    MAIN_LIBRARY,
    MMA,
    PRECISION,
    PROBE_VERSION,
    S1,
    SIMD_BARRIER,
    STAGE_FLOATS,
    TILE,
    VARIANTS,
    MetalTcError,
    library_sha256,
    library_source,
    require_size,
    require_uniform_barriers,
    threadgroup_bytes,
)


def test_each_variant_has_one_kernel_and_the_main_library_holds_the_probe():
    assert set(KERNELS) == {*KERNEL_NAME.values(), PROBE_VERSION}
    assert [KERNELS[KERNEL_NAME[v]].variant for v in VARIANTS] == list(VARIANTS)
    assert {KERNELS[name].library for name in KERNELS} == set(LIBRARIES)
    assert KERNELS[KERNEL_NAME[M3]].library == BFLOAT_LIBRARY
    assert "tc_m3_bfloat" not in library_source(MAIN_LIBRARY)
    assert library_source(BFLOAT_LIBRARY).count("kernel void ") == 1
    assert issubclass(MetalTcError, MetalFlashError)


def test_the_tiles_and_accumulators_are_the_same_in_every_kernel():
    s1 = KERNELS[KERNEL_NAME[S1]].source
    assert "simdgroup" not in s1
    assert "float acc[2][4];" in s1
    for variant in (M1, M2, M3):
        text = KERNELS[KERNEL_NAME[variant]].source
        kind = {"float32": "float", "float16": "half", "bfloat16": "bfloat"}[
            PRECISION[variant]
        ]
        assert f"threadgroup {kind} a_tile[{A_SLOTS}];" in text
        assert f"threadgroup {kind} b_tile[{B_SLOTS}];" in text
        assert "simdgroup_float8x8 acc[2][2];" in text
        assert f"simdgroup_matrix<{kind}, 8, 8> af[2];" in text
        assert text.count("simdgroup_multiply_accumulate(") == 1
        assert MMA[variant] is True
    for text in (KERNELS[KERNEL_NAME[v]].source for v in VARIANTS):
        assert "for (uint k0 = 0u; k0 < k; k0 += 16u) {" in text
        assert f"uint tiles_n = (n + {TILE - 1}u) / {TILE}u;" in text


def test_the_threadgroup_bytes_follow_the_padded_tiles_and_the_stage():
    assert threadgroup_bytes(S1) == (A_SLOTS + B_SLOTS) * 4 == 4864
    assert threadgroup_bytes(M1) == 4864 + STAGE_FLOATS * 4 == 5888
    assert threadgroup_bytes(M2) == threadgroup_bytes(M3) == 1216 * 2 + 1024
    for variant in VARIANTS:
        kernel = KERNELS[KERNEL_NAME[variant]]
        assert kernel.group_size == GROUP
        assert kernel.threadgroup_bytes == threadgroup_bytes(variant)


def test_every_barrier_is_counted_and_sits_where_the_rule_allows():
    for kernel in KERNELS.values():
        assert require_uniform_barriers(kernel.source) == kernel.barriers
    assert KERNELS[KERNEL_NAME[S1]].barriers == (2, 0)
    for variant in (M1, M2, M3):
        assert KERNELS[KERNEL_NAME[variant]].barriers == (2, 2)


@pytest.mark.parametrize(
    "text",
    [
        f"kernel void f() {{\n    {BARRIER}\n}}\n",
        "kernel void f() {\n    if (lane < 4u) {\n" f"        {BARRIER}\n    }}\n}}\n",
        "kernel void f() {\n    for (uint k0 = 0u; k0 < k; k0 += 16u) {\n"
        f"        {SIMD_BARRIER}\n    }}\n}}\n",
        f"kernel void f() {{\n    {SIMD_BARRIER}\n}}\n",
        "kernel void f() {\n",
        "}\n",
    ],
)
def test_a_barrier_in_a_thread_dependent_place_is_refused(text):
    with pytest.raises(MetalTcError):
        require_uniform_barriers(text)


def test_each_digest_hashes_the_exact_text():
    for kernel in KERNELS.values():
        assert kernel.source_sha256 == text_sha256(kernel.source)
        assert re.fullmatch(r"[0-9a-f]{64}", kernel.source_sha256)
        assert kernel.source.isascii()
        assert kernel.source in library_source(kernel.library)
    assert library_sha256(MAIN_LIBRARY) == text_sha256(library_source(MAIN_LIBRARY))
    assert library_sha256(MAIN_LIBRARY) != library_sha256(BFLOAT_LIBRARY)
    with pytest.raises(MetalTcError):
        library_source("other")


def test_a_kernel_record_refuses_a_wrong_count_name_or_library():
    kernel = KERNELS[KERNEL_NAME[M1]]
    for change in (
        {"barriers": (2, 0)},
        {"name": "tc_m9_float"},
        {"library": BFLOAT_LIBRARY},
        {"source": kernel.source.replace(SIMD_BARRIER, "")},
    ):
        with pytest.raises(MetalTcError):
            dataclasses.replace(kernel, **change)
    assert kernel.to_mapping()["precision"] == "float32"
    assert KERNELS[PROBE_VERSION].to_mapping()["precision"] is None


def test_sizes_are_bounded_with_this_labs_error():
    assert require_size(5, "k", 1, 8) == 5
    with pytest.raises(MetalTcError):
        require_size(9, "k", 1, 8)
    with pytest.raises(TypeError):
        require_size(True, "k", 1, 8)
