from fractions import Fraction

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_checklist_benchmark import modeled_loads_per_output
from ai_journey.matmul_checklist_bound_traffic import (
    BLOCKED,
    NO_REUSE,
    BlockTraffic,
    MatmulChecklistBoundError,
    NoReuseTraffic,
    device_bytes,
    intensity,
    kernel_traffic,
)
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.memory_access_tiling import MatmulShape
from test_matmul_checklist_config import TEST_CORRECTNESS

PINNED_SWEEP = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (2000, 3000, 1000),
]
SMALL = [(1, 1, 1), (17, 33, 9), (65, 68, 40), (31, 47, 129), (64, 48, 36)]


def walk(name, m, n, k):
    """Count by walking the staging and k loops of the kernel text, slot by slot.

    This follows the generated Metal source line by line: the scalar or float4
    staging of each phase with its guard, then ``min(BK, K - start)`` steps of
    TM + TN threadgroup reads and TM * TN fma for every thread of the group.
    """

    model = kernel_traffic(name, matmul_shape(m, n, k))
    bm, bn, bk = model.block_m, model.block_n, model.block_k
    tm, tn = model.thread_m, model.thread_n
    threads = (bm // tm) * (bn // tn)
    vector_a = model.vector_loads and k % 4 == 0
    vector_b = model.vector_loads and n % 4 == 0
    count = dict.fromkeys(
        ("a", "b", "operations", "slots", "slot_operations", "reads", "fma"), 0
    )
    for row0 in range(0, -(-m // bm) * bm, bm):
        for column0 in range(0, -(-n // bn) * bn, bn):
            for start in range(0, k, bk):
                width = 4 if vector_a else 1
                for e in range(bm * bk // width):
                    r, q = divmod(e, bk // width)
                    count["slots"] += width
                    count["slot_operations"] += 1
                    if row0 + r < m and start + width * q < k:
                        assert start + width * q + width <= k
                        count["a"] += width
                        count["operations"] += 1
                width = 4 if vector_b else 1
                for e in range(bk * bn // width):
                    q, s = divmod(e, bn // width)
                    count["slots"] += width
                    count["slot_operations"] += 1
                    if start + q < k and column0 + width * s < n:
                        assert column0 + width * s + width <= n
                        count["b"] += width
                        count["operations"] += 1
                steps = min(bk, k - start)
                count["reads"] += threads * steps * (tm + tn)
                count["fma"] += threads * steps * tm * tn
    return count


def test_s3_at_65_x_68_x_40_matches_the_hand_count():
    # 2 x 2 groups of 64 x 64, 3 phases of 16 (16, 16 and 8 steps).
    model = kernel_traffic("matmul_s3_block_4x4", matmul_shape(65, 68, 40))
    mapping = model.to_mapping()

    assert (model.block_rows, model.block_columns, model.phases) == (2, 2, 3)
    assert model.launched_threads == 4 * 256
    assert model.a_loads == 2 * 65 * 40 == 5200
    assert model.b_loads == 2 * 40 * 68 == 5440
    assert model.a_slots + model.b_slots == (64 * 16 + 16 * 64) * 3 * 4 == 24576
    assert model.padded_slots == 24576 - 10640 == 13936
    assert model.load_operations == 10640
    assert device_bytes(model) == 4 * (10640 + 65 * 68) == 60240
    assert intensity(model) == Fraction(2 * 65 * 68 * 40, 60240) == Fraction(4420, 753)
    assert model.executed_fma == 4 * 64 * 64 * 40 == 655360
    assert model.threadgroup_read_elements == 1024 * 40 * 8 == 327680
    assert mapping["threadgroup"] == {
        "store_elements": 24576,
        "store_operations": 24576,
        "read_elements": 327680,
        "reads_per_fma": {"numerator": 1, "denominator": 2, "decimal": "0.500000"},
        "barriers_per_thread": 6,
        "fma_per_thread_per_phase": 256,
    }
    assert mapping["device"]["loads_per_output"] == {
        "numerator": 532,
        "denominator": 221,
        "decimal": "2.407240",
    }


def test_float4_staging_moves_the_same_elements_in_a_quarter_of_the_operations():
    scalar = kernel_traffic("matmul_s3_block_4x4", matmul_shape(65, 68, 40))
    vector = kernel_traffic("matmul_s4_vec4", matmul_shape(65, 68, 40))

    assert (vector.vector_a, vector.vector_b) == (True, True)
    assert (vector.a_loads, vector.b_loads) == (scalar.a_loads, scalar.b_loads)
    assert vector.load_operations == 5200 // 4 + 5440 // 4 == 2660
    assert vector.threadgroup_store_operations == 24576 // 4
    assert device_bytes(vector) == device_bytes(scalar)
    # K = 9 and N = 33 keep both operands on the scalar path.
    ragged = kernel_traffic("matmul_s4_vec4", matmul_shape(17, 33, 9))
    assert (ragged.vector_a, ragged.vector_b) == (False, False)
    assert ragged.load_operations == ragged.a_loads + ragged.b_loads
    # K = 129 is odd, but N = 48 still stages B with float4 values.
    mixed = kernel_traffic("matmul_s5_pad4", matmul_shape(31, 48, 129))
    assert (mixed.vector_a, mixed.vector_b) == (False, True)
    assert mixed.load_operations == mixed.a_loads + mixed.b_loads // 4


@pytest.mark.parametrize(
    "name", [name for name, variant in VARIANTS.items() if variant.group_size]
)
@pytest.mark.parametrize(("m", "n", "k"), SMALL)
def test_every_block_count_equals_a_walk_of_the_kernel_loops(name, m, n, k):
    model = kernel_traffic(name, matmul_shape(m, n, k))
    count = walk(name, m, n, k)

    assert (model.a_loads, model.b_loads) == (count["a"], count["b"])
    assert model.load_operations == count["operations"]
    assert model.a_slots + model.b_slots == count["slots"]
    assert model.threadgroup_store_operations == count["slot_operations"]
    assert model.threadgroup_read_elements == count["reads"]
    assert model.executed_fma == count["fma"]


def test_the_steps_change_the_threadgroup_reads_that_the_hypotheses_name():
    shape = matmul_shape(1024, 1024, 1024)
    reads = {
        name: kernel_traffic(name, shape).reads_per_fma
        for name in VARIANTS
        if VARIANTS[name].group_size is not None
    }

    # S2: from 2 to 1.25 for each fma; S3: from 1.25 to 0.5.
    assert reads["matmul_tiled_32"] == 2
    assert reads["matmul_s2_coarse_4x1"] == Fraction(5, 4)
    assert {reads[name] for name in list(reads)[2:]} == {Fraction(1, 2)}


def test_the_pinned_intensities_are_the_values_that_the_plan_quotes():
    shape = matmul_shape(4096, 4096, 4096)

    def value(name):
        return round(float(intensity(kernel_traffic(name, shape))), 2)

    # The S4 hypothesis quotes 15.9 FLOPs for each byte for S3 at 4096 cubed.
    assert value("matmul_s3_block_4x4") == value("matmul_s7_interior") == 15.88
    assert value("matmul_s6_group_8x8") == value("matmul_tiled_32") == 7.97
    assert value("matmul_s6_group_32x32") == 31.51
    assert value("matmul_x_columns") == 0.25


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize(
    "shape", [*PINNED_SWEEP, *SMALL, *(tuple(item) for item in TEST_CORRECTNESS)]
)
def test_loads_per_output_equal_the_model_numbers_of_the_checklist_report(name, shape):
    checked = matmul_shape(*shape)
    model = kernel_traffic(name, checked)
    mapping = model.to_mapping()["device"]["loads_per_output"]

    ratio = Fraction(mapping["numerator"], mapping["denominator"])
    assert float(ratio) == modeled_loads_per_output(VARIANTS[name], checked)


@pytest.mark.parametrize("shape", [*PINNED_SWEEP, *SMALL])
def test_the_day_74_kernels_and_tile_32_agree_with_the_day_74_models(shape):
    checked = matmul_shape(*shape)
    naive = kernel_traffic("matmul_x_rows", checked)
    tiled = kernel_traffic("matmul_tiled_32", checked)

    assert isinstance(naive, NoReuseTraffic) and naive.day_74_check()
    assert isinstance(tiled, BlockTraffic) and tiled.day_74_check()
    assert naive.to_mapping()["tiles"] is None
    assert naive.to_mapping()["threadgroup"] is None
    assert naive.to_mapping()["model"] == NO_REUSE
    assert tiled.to_mapping()["model"] == BLOCKED
    with pytest.raises(MatmulChecklistBoundError, match="Day 74 count"):
        kernel_traffic("matmul_s3_block_4x4", checked).day_74_check()


def test_every_mapping_holds_the_same_fields_for_both_models():
    shape = matmul_shape(17, 33, 9)
    naive = kernel_traffic("matmul_x_columns", shape).to_mapping()
    block = kernel_traffic("matmul_s7_interior", shape).to_mapping()

    assert set(naive) == set(block)
    assert set(naive["device"]) == set(block["device"])
    assert naive["device"]["load_operations"] == 2 * 17 * 33 * 9
    assert naive["executed_fma"] == 17 * 33 * 9
    assert block["block"] == [64, 64, 16] and block["outputs_per_thread"] == [4, 4]
    assert block["vector_paths"] == {"a": False, "b": False}


def test_invalid_models_are_refused_before_any_count():
    shape = matmul_shape(8, 8, 8)
    with pytest.raises(MatmulChecklistBoundError, match="registered kernel"):
        kernel_traffic("matmul_tiled_16", shape)
    with pytest.raises(TypeError, match="MatmulShape"):
        kernel_traffic("matmul_x_rows", (8, 8, 8))
    with pytest.raises(TypeError, match="MatmulShape"):
        NoReuseTraffic(MatmulShape(8, 8, 8).to_mapping())
    with pytest.raises(MatmulChecklistBoundError, match="whole thread blocks"):
        BlockTraffic(shape, 30, 32, 16, 4, 4, False)
    with pytest.raises(MatmulChecklistBoundError, match="exceeds 1024"):
        BlockTraffic(shape, 128, 128, 16, 2, 2, False)
    with pytest.raises(MatmulChecklistBoundError, match="multiples of 4"):
        BlockTraffic(shape, 64, 64, 18, 4, 4, True)
    with pytest.raises(TypeError, match="bool"):
        BlockTraffic(shape, 64, 64, 16, 4, 4, 1)
    with pytest.raises(KernelMetalError):
        BlockTraffic(shape, 64, 64, 0, 4, 4, False)
