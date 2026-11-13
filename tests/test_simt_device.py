from dataclasses import replace

import pytest

from ai_journey.simt_device import (
    PINNED_ARCHITECTURES,
    PINNED_DEVICES,
    SimtArchitecture,
    SimtDevice,
    SimtError,
)

KIB = 1024


def test_pinned_architectures_match_the_programming_guide_tables():
    # CUDA C++ Programming Guide v12.6, Table 21, columns 7.0, 7.5 and 8.0.
    expected = {
        "7.0": (2048, 64, 32, 96 * KIB, 96 * KIB),
        "7.5": (1024, 32, 16, 64 * KIB, 64 * KIB),
        "8.0": (2048, 64, 32, 164 * KIB, 163 * KIB),
    }
    assert sorted(PINNED_ARCHITECTURES) == sorted(expected)
    for capability, values in expected.items():
        architecture = PINNED_ARCHITECTURES[capability]
        threads, warps, blocks, per_sm, optin = values
        assert architecture.compute_capability == capability
        assert architecture.warp_size == 32
        assert architecture.max_threads_per_block == 1024
        assert (
            architecture.max_block_dim_x,
            architecture.max_block_dim_y,
            architecture.max_block_dim_z,
        ) == (1024, 1024, 64)
        assert (
            architecture.max_grid_dim_x,
            architecture.max_grid_dim_y,
            architecture.max_grid_dim_z,
        ) == (2**31 - 1, 65_535, 65_535)
        assert architecture.max_threads_per_sm == threads
        assert architecture.max_warps_per_sm == warps
        assert architecture.max_blocks_per_sm == blocks
        assert architecture.registers_per_sm == 65_536
        assert architecture.registers_per_block == 65_536
        assert architecture.max_registers_per_thread == 255
        assert architecture.shared_memory_per_sm_bytes == per_sm
        assert architecture.shared_memory_per_block_optin_bytes == optin
        assert architecture.shared_memory_per_block_bytes == 48 * KIB
        assert architecture.fp32_lanes_per_sm == 64
        assert architecture.warp_schedulers_per_sm == 4


def test_pinned_shared_memory_and_register_allocation_rules_follow_sources():
    volta, turing, ampere = (PINNED_ARCHITECTURES[key] for key in ("7.0", "7.5", "8.0"))

    # Guide sections 16.6.4 and 16.7.3 list the configurable capacities.
    assert volta.shared_memory_carveouts_bytes == tuple(
        size * KIB for size in (0, 8, 16, 32, 64, 96)
    )
    assert turing.shared_memory_carveouts_bytes == (32 * KIB, 64 * KIB)
    assert ampere.shared_memory_carveouts_bytes == tuple(
        size * KIB for size in (0, 8, 16, 32, 64, 100, 132, 164)
    )
    # 164 KB per SM minus the 163 KB block maximum is the 1 KB system reserve.
    assert ampere.reserved_shared_memory_per_block_bytes == 1 * KIB
    assert volta.reserved_shared_memory_per_block_bytes == 0
    assert turing.reserved_shared_memory_per_block_bytes == 0
    # cuda_occupancy.h: 256-byte shared allocation units on 7.x, 128 on 8.x.
    assert volta.shared_memory_allocation_unit_bytes == 256
    assert turing.shared_memory_allocation_unit_bytes == 256
    assert ampere.shared_memory_allocation_unit_bytes == 128
    for architecture in (volta, turing, ampere):
        assert architecture.register_allocation_unit == 256
        assert architecture.register_allocation_granularity == "warp"
        assert architecture.register_sub_partitions_per_sm == 4
        assert any("cuda_occupancy.h" in source for source in architecture.sources)
        assert any("Table 21" in source for source in architecture.sources)


def test_derived_fp32_peaks_match_each_datasheet_within_published_rounding():
    # V100 SXM2: 80 SMs * 64 lanes * 2 FLOPs * 1.530 GHz = 15.6672 TFLOPS,
    #   |15.6672 - 15.7| = 0.0328 <= 0.05 (datasheet prints one decimal).
    # T4:        40 SMs * 64 lanes * 2 FLOPs * 1.590 GHz =  8.1408 TFLOPS,
    #   |8.1408 - 8.1|   = 0.0408 <= 0.05.
    # A100 SXM4: 108 SMs * 64 lanes * 2 FLOPs * 1.410 GHz = 19.49184 TFLOPS,
    #   |19.49184 - 19.5| = 0.00816 <= 0.05.
    expected = {
        "tesla-v100-sxm2": ("15.7", 15_667_200_000_000, 32_800_000_000),
        "tesla-t4": ("8.1", 8_140_800_000_000, 40_800_000_000),
        "a100-sxm4-80gb": ("19.5", 19_491_840_000_000, 8_160_000_000),
    }
    assert sorted(PINNED_DEVICES) == sorted(expected)
    for key, (published, derived, difference) in expected.items():
        device = PINNED_DEVICES[key]
        check = device.fp32_datasheet_check()
        assert device.peak_fp32_flops_per_second == derived
        assert check.published_tflops == published
        assert check.published_flops_per_second == int(published.replace(".", "")) * (
            10**11
        )
        assert check.rounding_half_unit_flops_per_second == 50_000_000_000
        assert check.absolute_difference_flops_per_second == difference
        assert check.within_published_rounding is True


def test_pinned_devices_declare_datasheet_bandwidth_in_decimal_bytes():
    assert PINNED_DEVICES["tesla-v100-sxm2"].dram_bandwidth_bytes_per_second == (
        900 * 10**9
    )
    assert PINNED_DEVICES["tesla-t4"].dram_bandwidth_bytes_per_second == 300 * 10**9
    assert PINNED_DEVICES["a100-sxm4-80gb"].dram_bandwidth_bytes_per_second == (
        2_039 * 10**9
    )
    assert PINNED_DEVICES["a100-sxm4-80gb"].compute_capability == "8.0"
    assert any("320 GB/sec" in source for source in PINNED_DEVICES["tesla-t4"].sources)


def test_device_rejects_specs_that_do_not_reproduce_the_datasheet_peak():
    v100 = PINNED_DEVICES["tesla-v100-sxm2"]

    # 80 * 64 * 2 * 1.380 GHz = 14.1312 TFLOPS is 1.57 TFLOPS away from 15.7.
    with pytest.raises(SimtError, match="published rounding"):
        replace(v100, boost_clock_mhz=1380)
    # A finer published figure tightens the tolerance to 0.005 TFLOPS.
    with pytest.raises(SimtError, match="published rounding"):
        replace(v100, datasheet_fp32_tflops="15.70")


def test_architecture_and_device_mappings_round_trip_with_exact_fields():
    ampere = PINNED_ARCHITECTURES["8.0"]
    a100 = PINNED_DEVICES["a100-sxm4-80gb"]

    assert SimtArchitecture.from_mapping(ampere.to_mapping()) == ampere
    assert SimtDevice.from_mapping(a100.to_mapping(), PINNED_ARCHITECTURES) == a100
    assert ampere.to_mapping()["shared_memory_carveouts_bytes"][-1] == 164 * KIB

    extra = {**ampere.to_mapping(), "l2_cache_bytes": 1}
    with pytest.raises(SimtError, match="extra=\\['l2_cache_bytes'\\]"):
        SimtArchitecture.from_mapping(extra)
    missing = a100.to_mapping()
    del missing["sm_count"]
    with pytest.raises(SimtError, match="fields do not match"):
        SimtDevice.from_mapping(missing, PINNED_ARCHITECTURES)
    unknown = {**a100.to_mapping(), "compute_capability": "9.0"}
    with pytest.raises(SimtError, match="not a declared architecture"):
        SimtDevice.from_mapping(unknown, PINNED_ARCHITECTURES)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"warp_size": True}, TypeError, "warp_size must be an integer"),
        ({"compute_capability": "8"}, SimtError, "look like"),
        ({"max_warps_per_sm": 48}, SimtError, "max_warps_per_sm \\* warp_size"),
        ({"max_block_dim_z": 2048}, SimtError, "block dimension limits"),
        ({"registers_per_block": 131_072}, SimtError, "cannot exceed registers"),
        ({"register_allocation_unit": 200}, SimtError, "power of two"),
        ({"register_allocation_granularity": "thread"}, SimtError, "granularity"),
        (
            {"shared_memory_carveouts_bytes": (0, 64 * KIB, 32 * KIB, 164 * KIB)},
            SimtError,
            "strictly increasing",
        ),
        (
            {"shared_memory_carveouts_bytes": [0, 164 * KIB]},
            SimtError,
            "nonempty tuple",
        ),
        (
            {"shared_memory_carveouts_bytes": (0, 100, 164 * KIB)},
            SimtError,
            "whole allocation units",
        ),
        (
            {"shared_memory_per_sm_bytes": 228 * KIB},
            SimtError,
            "largest carveout",
        ),
        (
            {"shared_memory_per_block_optin_bytes": 164 * KIB},
            SimtError,
            "opt-in block and its reservation",
        ),
        ({"sources": ()}, SimtError, "one to eight citations"),
        ({"sources": ("table", "table")}, SimtError, "repeat a citation"),
        ({"sources": (" padded",)}, SimtError, "canonical printable ASCII"),
    ],
)
def test_architecture_rejects_inconsistent_or_ambiguous_limits(changes, error, message):
    with pytest.raises(error, match=message):
        replace(PINNED_ARCHITECTURES["8.0"], **changes)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"key": "A100"}, SimtError, "lowercase slug"),
        ({"name": ""}, SimtError, "canonical printable ASCII"),
        ({"sm_count": 108.0}, TypeError, "sm_count must be an integer"),
        ({"dram_bandwidth_gb_per_second": 0}, SimtError, "between 1"),
        ({"datasheet_fp32_tflops": "19.5e0"}, SimtError, "canonical decimal"),
        ({"architecture": "8.0"}, TypeError, "SimtArchitecture"),
    ],
)
def test_device_rejects_invalid_product_fields(changes, error, message):
    with pytest.raises(error, match=message):
        replace(PINNED_DEVICES["a100-sxm4-80gb"], **changes)


def test_pinned_catalogs_are_read_only():
    with pytest.raises(TypeError):
        PINNED_DEVICES["h100-sxm5"] = PINNED_DEVICES["a100-sxm4-80gb"]
    with pytest.raises(TypeError):
        PINNED_ARCHITECTURES["9.0"] = PINNED_ARCHITECTURES["8.0"]
