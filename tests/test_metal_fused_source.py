import re

import numpy
import pytest

from ai_journey.matmul_tiled_source import BARRIER, HEADER
from ai_journey.matmul_tiled_variants import text_sha256
from ai_journey.metal_fused_source import (
    EXP_PROBE,
    F1,
    F2,
    F3,
    KERNELS,
    LOWEST,
    LOWEST_TEXT,
    MAX_ROW_LENGTH,
    SLOTS,
    STEP_VARIANTS,
    U0,
    U0_EXP_SUM,
    U0_NORMALIZE,
    U0_ROW_MAX,
    VARIANTS,
    FusedKernel,
    MetalFusedError,
    SoftmaxVariant,
    fused_library,
    group_size,
    library_sha256,
    require_uniform_barriers,
    slots_used,
)

_ARRAY = re.compile(r"threadgroup float (\w+)\[(\d+)\];")


def test_the_library_joins_every_kernel_under_one_header():
    library = fused_library()

    assert library.startswith(HEADER)
    assert library.isascii()
    assert re.findall(r"kernel void (\w+)\(", library) == list(KERNELS)
    assert library_sha256() == text_sha256(library)
    for kernel in KERNELS.values():
        assert kernel.source in library
        assert kernel.source_sha256 == text_sha256(kernel.source)


def test_the_steps_hold_their_kernels_in_launch_order():
    assert tuple(VARIANTS) == STEP_VARIANTS == (U0, F1, F2, F3)
    assert VARIANTS[U0].kernels == (U0_ROW_MAX, U0_EXP_SUM, U0_NORMALIZE)
    assert [VARIANTS[name].launches_per_call for name in STEP_VARIANTS] == [
        3,
        1,
        1,
        1,
    ]
    assert [VARIANTS[name].row_reads for name in STEP_VARIANTS] == [3, 1, 1, 1]
    assert [VARIANTS[name].step for name in STEP_VARIANTS] == ["U0", "F1", "F2", "F3"]
    for variant in VARIANTS.values():
        assert variant.source == "\n".join(
            KERNELS[name].source for name in variant.kernels
        )
        assert variant.source_sha256 == text_sha256(variant.source)
        assert variant.to_mapping()["source_sha256"] == variant.source_sha256
    assert EXP_PROBE not in {name for v in VARIANTS.values() for name in v.kernels}


def test_each_barrier_is_outside_every_thread_condition():
    counts = {name: require_uniform_barriers(k.source) for name, k in KERNELS.items()}

    # One barrier after the store and one in the tree loop per reduced value.
    assert counts == {
        U0_ROW_MAX: 2,
        U0_EXP_SUM: 2,
        U0_NORMALIZE: 0,
        F1: 4,
        F2: 2,
        F3: 1,
        EXP_PROBE: 0,
    }
    for name in (F1, F2, F3):
        assert KERNELS[name].source.count(BARRIER) == counts[name]


@pytest.mark.parametrize(
    "opener",
    [
        "if (lane == 0u) {",
        "for (uint c = lane; c < length; c += size) {",
        "for (uint i = 0u; i < 16u; ++i) {",
    ],
)
def test_the_barrier_check_refuses_a_barrier_inside_a_thread_condition(opener):
    text = "\n".join(
        [
            "kernel void softmax_bad(device float* y [[buffer(0)]]) {",
            f"    {opener}",
            f"        {BARRIER}",
            "    }",
            "}",
        ]
    )

    with pytest.raises(MetalFusedError, match="depends on the thread"):
        require_uniform_barriers(text)


def test_the_barrier_check_refuses_unbalanced_blocks():
    with pytest.raises(MetalFusedError, match="not closed"):
        require_uniform_barriers("kernel void softmax_bad() {\n    if (x) {\n}")
    with pytest.raises(MetalFusedError, match="no open block"):
        require_uniform_barriers("}\n")


def test_each_declared_threadgroup_size_is_the_arrays_of_its_text():
    for name, kernel in KERNELS.items():
        arrays = _ARRAY.findall(kernel.source)
        assert sum(4 * int(count) for _, count in arrays) == kernel.threadgroup_bytes
        assert len({array for array, _ in arrays}) == len(arrays), name


def test_simd_functions_appear_only_where_declared():
    for name, kernel in KERNELS.items():
        uses = bool(re.search(r"\bsimd_(max|sum)\(", kernel.source))
        assert uses is kernel.simd_functions, name
        attributes = "[[thread_index_in_simdgroup]]" in kernel.source
        assert attributes is kernel.simd_functions, name
    assert [VARIANTS[name].simd_functions for name in STEP_VARIANTS] == [
        False,
        False,
        True,
        True,
    ]


def _code(text):
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )


def test_the_math_is_precise_and_never_uses_an_infinity():
    code = _code(fused_library())

    assert "INFINITY" not in code and "isinf" not in code
    assert re.findall(r"(?<!precise::)\bexp\(", code) == []
    assert re.findall(r"(?<!precise::)\bdivide\(", code) == []
    # The only divisions are the integer ones of the group size.
    assert set(re.findall(r"\S+ / \S+", code)) == {"size / 2u;", "size / 32u;"}
    # F3 calls exp in its online pass, its two merges and its write pass.
    assert {
        name: _code(kernel.source).count("precise::exp(")
        for name, kernel in KERNELS.items()
    } == {
        U0_ROW_MAX: 0,
        U0_EXP_SUM: 1,
        U0_NORMALIZE: 1,
        F1: 1,
        F2: 1,
        F3: 4,
        EXP_PROBE: 1,
    }
    assert LOWEST_TEXT in KERNELS[F1].source
    assert float(numpy.float32(LOWEST_TEXT.removesuffix("f"))) == LOWEST
    # The float32 nearest to the text is the largest finite magnitude.
    assert (
        numpy.float32(float(LOWEST_TEXT.removesuffix("f")))
        == numpy.finfo(numpy.float32).min
    )


def test_the_online_step_rescales_its_running_sum():
    text = KERNELS[F3].source

    assert "float scale = precise::exp(min(m, v[i]) - max(m, v[i]));" in text
    assert "d = v[i] > m ? fma(d, scale, 1.0f) : d + scale;" in text
    assert "float simd_d = simd_sum(d * precise::exp(m - simd_m));" in text
    assert "float total = simd_sum(group_d * precise::exp(group_m - row_max));" in text


def test_the_fused_steps_read_each_slot_once_and_write_it_once():
    for name in (F1, F2, F3):
        text = KERNELS[name].source
        assert text.count("x[start + c]") == 1, name
        assert text.count("y[start + c] =") == 1, name
    assert (
        sum(KERNELS[name].source.count("x[start + c]") for name in VARIANTS[U0].kernels)
        == 3
    )


@pytest.mark.parametrize(
    ("longest", "size"),
    [
        (1, 32),
        (31, 32),
        (33, 32),
        (512, 32),
        (513, 64),
        (1000, 64),
        (2049, 256),
        (4096, 256),
        (8192, 512),
        (16384, 1024),
    ],
)
def test_the_group_size_is_the_fewest_threads_whose_slots_hold_the_row(longest, size):
    assert group_size(longest) == size
    assert size * SLOTS >= longest
    assert size == 32 or (size // 2) * SLOTS < longest


def test_rows_past_the_slots_of_the_largest_group_are_refused():
    assert MAX_ROW_LENGTH == 16384
    with pytest.raises(MetalFusedError, match="longest row"):
        group_size(MAX_ROW_LENGTH + 1)
    with pytest.raises(MetalFusedError, match="longest row"):
        group_size(0)
    with pytest.raises(TypeError):
        group_size(True)
    assert slots_used(33, 32) == 2
    assert slots_used(1, 1024) == 1
    with pytest.raises(MetalFusedError, match="more than 16 slots"):
        slots_used(513, 32)


def test_kernel_and_variant_records_validate_their_fields():
    with pytest.raises(MetalFusedError, match="softmax_"):
        FusedKernel("matmul_x", ("y",), 0, False, "kernel void matmul_x(")
    with pytest.raises(MetalFusedError, match="does not define"):
        FusedKernel("softmax_x", ("y",), 0, False, "kernel void other(")
    with pytest.raises(TypeError):
        FusedKernel("softmax_x", ("y",), 0, 1, "kernel void softmax_x(")
    with pytest.raises(MetalFusedError, match="registered step"):
        SoftmaxVariant("F9", "softmax_f9", (F1,), "threadgroup_tree", 1, "x")
    with pytest.raises(MetalFusedError, match="reduction"):
        SoftmaxVariant("F1", F1, (F1,), "atomic", 1, "x")
    assert KERNELS[U0_ROW_MAX].to_mapping() == {
        "name": U0_ROW_MAX,
        "buffers": ["row_max", "x", "starts", "lengths"],
        "threadgroup_bytes": 4096,
        "simd_functions": False,
        "source_sha256": KERNELS[U0_ROW_MAX].source_sha256,
    }
