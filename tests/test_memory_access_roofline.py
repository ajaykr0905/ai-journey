import pytest

from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_roofline import (
    compulsory_workload,
    roofline_estimate_mapping,
    sweep_tile_rooflines,
    traffic_workload,
)
from ai_journey.memory_access_tiling import (
    MatmulShape,
    NaiveMatmulTraffic,
    TiledMatmulTraffic,
)
from ai_journey.roofline import KernelWorkload, RooflineHardware


def ceilings(peak: float, bandwidth: float) -> RooflineHardware:
    return RooflineHardware(
        peak_flops_per_second=peak, memory_bandwidth_bytes_per_second=bandwidth
    )


def test_traffic_models_become_roofline_workloads_with_global_bytes():
    shape = MatmulShape(64, 64, 64)

    assert traffic_workload(NaiveMatmulTraffic(shape)) == KernelWorkload(
        kind="naive_global_matmul",
        flops=524_288,
        bytes_read=2_097_152,
        bytes_written=16_384,
    )
    assert traffic_workload(TiledMatmulTraffic(shape, tile=32)) == KernelWorkload(
        kind="tiled_global_matmul_t32",
        flops=524_288,
        bytes_read=65_536,
        bytes_written=16_384,
    )
    assert compulsory_workload(shape) == KernelWorkload(
        kind="matmul",
        flops=524_288,
        bytes_read=32_768,
        bytes_written=16_384,
    )


def test_sweep_classifies_naive_tiled_and_compulsory_traffic_on_one_roofline():
    sweep = sweep_tile_rooflines(
        MatmulShape(64, 64, 64), tiles=(8, 16, 32), hardware=ceilings(40, 10)
    )

    assert sweep.hardware.ridge_point == 4
    assert sweep.naive.bottleneck == "memory"
    assert sweep.compulsory.bottleneck == "compute"
    assert [item.estimate.bottleneck for item in sweep.tiles] == [
        "memory",
        "memory",
        "compute",
    ]
    assert [item.traffic.tile for item in sweep.tiles] == [8, 16, 32]
    assert sweep.bottleneck_counts == {"compute": 1, "memory": 2, "ridge": 0}
    assert sweep.first_compute_bound_tile == 32
    tiled = sweep.tiles[2].estimate
    assert tiled.workload.arithmetic_intensity == 524_288 / 81_920
    assert tiled.compute_seconds == 524_288 / 40
    assert tiled.memory_seconds == 81_920 / 10
    assert tiled.lower_bound_seconds == tiled.compute_seconds


def test_intensity_grows_with_tile_width_but_stays_below_compulsory():
    sweep = sweep_tile_rooflines(
        MatmulShape(256, 256, 256), tiles=(1, 8, 16, 32), hardware=ceilings(1e3, 1e2)
    )
    intensities = [item.estimate.workload.arithmetic_intensity for item in sweep.tiles]

    assert intensities == sorted(intensities)
    assert intensities[0] == sweep.naive.workload.arithmetic_intensity
    assert intensities[-1] < sweep.compulsory.workload.arithmetic_intensity
    assert sweep.first_compute_bound_tile is None


def test_declared_ridge_equality_is_reported_explicitly():
    shape = MatmulShape(64, 64, 64)
    workload = traffic_workload(TiledMatmulTraffic(shape, tile=32))
    sweep = sweep_tile_rooflines(
        shape,
        tiles=(32,),
        hardware=ceilings(workload.flops, workload.total_bytes),
    )

    assert sweep.tiles[0].estimate.compute_seconds == 1.0
    assert sweep.tiles[0].estimate.memory_seconds == 1.0
    assert sweep.tiles[0].estimate.bottleneck == "ridge"
    assert sweep.bottleneck_counts["ridge"] == 1
    assert sweep.first_compute_bound_tile is None


def test_estimate_mapping_is_json_safe_and_complete():
    sweep = sweep_tile_rooflines(
        MatmulShape(64, 64, 64), tiles=(16,), hardware=ceilings(40, 10)
    )
    mapping = roofline_estimate_mapping(sweep.tiles[0].estimate)

    assert mapping["kind"] == "tiled_global_matmul_t16"
    assert mapping["total_bytes"] == 147_456
    assert mapping["bottleneck"] == "memory"
    assert mapping["lower_bound_seconds"] == 147_456 / 10
    assert set(mapping) == {
        "kind",
        "flops",
        "bytes_read",
        "bytes_written",
        "total_bytes",
        "arithmetic_intensity",
        "compute_seconds",
        "memory_seconds",
        "lower_bound_seconds",
        "attainable_flops_per_second",
        "bottleneck",
    }


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: traffic_workload("naive"), TypeError, "traffic"),
        (lambda: compulsory_workload((1, 1, 1)), TypeError, "MatmulShape"),
        (
            lambda: compulsory_workload(MatmulShape(1, 1, 1), "int8"),
            MemoryAccessError,
            "dtype",
        ),
        (
            lambda: sweep_tile_rooflines(
                MatmulShape(8, 8, 8), tiles=(), hardware=ceilings(1, 1)
            ),
            MemoryAccessError,
            "tiles",
        ),
        (
            lambda: sweep_tile_rooflines(
                MatmulShape(8, 8, 8), tiles=(16, 8), hardware=ceilings(1, 1)
            ),
            MemoryAccessError,
            "increasing",
        ),
        (
            lambda: sweep_tile_rooflines(
                MatmulShape(8, 8, 8), tiles=(8, 8), hardware=ceilings(1, 1)
            ),
            MemoryAccessError,
            "unique",
        ),
        (
            lambda: sweep_tile_rooflines(
                MatmulShape(8, 8, 8), tiles=[8], hardware=ceilings(1, 1)
            ),
            MemoryAccessError,
            "tuple",
        ),
        (
            lambda: sweep_tile_rooflines(
                MatmulShape(8, 8, 8), tiles=(8,), hardware=(1, 1)
            ),
            TypeError,
            "RooflineHardware",
        ),
        (lambda: roofline_estimate_mapping({}), TypeError, "RooflineEstimate"),
    ],
)
def test_roofline_sweep_rejects_ambiguous_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()
