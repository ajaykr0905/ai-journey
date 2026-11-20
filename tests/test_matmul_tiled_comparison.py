import pytest

from ai_journey.kernel_metal_benchmark import DeviceTiming
from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits
from ai_journey.matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_benchmark import TiledPoint
from ai_journey.matmul_tiled_comparison import (
    CUBLAS,
    FASTER,
    NO_CLEAR_DIFFERENCE,
    RATIO_RULE,
    SLOWER,
    TILED_RULE,
    comparison,
    three_numbers,
    vendor_label,
)
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL, SideCheck
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import VARIANTS

NAMES = tuple(VARIANTS)
TILES = {name: variant.tile for name, variant in VARIANTS.items()}
SHAPE = matmul_shape(64, 64, 64)


def synthetic_point(samples, variants=tuple(VARIANTS.values())):
    """Build a point mapping whose sides took the given sample times."""

    launches = [
        CheckedLaunch(
            matmul_launch(X_COLUMNS, SHAPE, (32, 8)), PipelineLimits(1024, 32, 0)
        )
    ]
    launches += [
        CheckedLaunch(
            matmul_launch(X_COLUMNS, SHAPE, variant.group_size),
            PipelineLimits(1024, 32, variant.threadgroup_bytes),
        )
        for variant in variants
    ]
    timings = tuple(
        DeviceTiming(name, 1, 0, 3, tuple(series), "injected")
        for name, series in samples.items()
    )
    checks = {name: SideCheck(True, 0.0) for name in list(samples)[1:]}
    return TiledPoint(0, SHAPE, variants, tuple(launches), timings, checks).to_mapping()


def numbers(samples, ratio=1.1):
    return three_numbers(synthetic_point(samples), NAMES, TILES, ratio)


def test_the_fastest_admitted_tile_is_the_tiled_number_and_steps_use_one_run():
    result = numbers(
        {
            NAIVE_KERNEL: [1000, 1010, 1020],
            "matmul_tiled_8": [1400, 1410, 1420],
            "matmul_tiled_16": [1200, 1210, 1220],
            "matmul_tiled_32": [800, 810, 820],
            TORCH_MATMUL: [100, 101, 102],
        }
    )

    flops = SHAPE.flops
    assert result["naive"] == {
        "side": NAIVE_KERNEL,
        "median_gflops_per_second": flops / 1010,
        "iqr_gflops_per_second": [flops / 1020, flops / 1000],
    }
    assert result["tiled"]["side"] == "matmul_tiled_32"
    assert result["tiled"]["tile"] == 32
    assert result["vendor"]["median_gflops_per_second"] == flops / 101
    assert result["cublas"] == "not measured, no NVIDIA GPU or cuBLAS on this host"
    assert result["tiled_over_naive"] == {
        "first": "matmul_tiled_32",
        "second": NAIVE_KERNEL,
        "ratio": (flops / 810) / (flops / 1010),
        "iqr_separated": True,
        "verdict": FASTER,
    }
    assert result["vendor_over_tiled"]["verdict"] == FASTER
    assert result["vendor_over_tiled"]["first"] == TORCH_MATMUL


def test_a_tie_goes_to_the_first_listed_tile_and_a_slower_tile_is_named_slower():
    result = numbers(
        {
            NAIVE_KERNEL: [1000, 1000, 1000],
            "matmul_tiled_8": [2000, 2000, 2000],
            "matmul_tiled_16": [2000, 2000, 2000],
            "matmul_tiled_32": [2000, 2000, 2000],
            TORCH_MATMUL: [100, 100, 100],
        }
    )

    assert result["tiled"]["side"] == "matmul_tiled_8"
    assert result["tiled_over_naive"]["ratio"] == 0.5
    assert result["tiled_over_naive"]["verdict"] == SLOWER


@pytest.mark.parametrize(
    ("naive", "tiled", "separated"),
    [
        ([1000, 1010, 1500], [800, 900, 1200], False),
        ([1000, 1000, 1000], [950, 950, 950], True),
    ],
)
def test_a_small_ratio_or_overlapping_ranges_name_no_clear_difference(
    naive, tiled, separated
):
    result = numbers(
        {
            NAIVE_KERNEL: naive,
            "matmul_tiled_8": [5000] * 3,
            "matmul_tiled_16": [5000] * 3,
            "matmul_tiled_32": tiled,
            TORCH_MATMUL: [100, 100, 100],
        }
    )

    assert result["tiled_over_naive"]["verdict"] == NO_CLEAR_DIFFERENCE
    assert result["tiled_over_naive"]["iqr_separated"] is separated


def test_with_every_tile_refused_there_is_no_tiled_number_and_no_step():
    point = synthetic_point(
        {NAIVE_KERNEL: [1000, 1000, 1000], TORCH_MATMUL: [100, 100, 100]},
        variants=(),
    )

    result = three_numbers(point, NAMES, TILES, 1.1)

    assert result["tiled"] is None
    assert result["tiled_over_naive"] is None and result["vendor_over_tiled"] is None
    assert result["vendor"]["side"] == TORCH_MATMUL
    with pytest.raises(MatmulTiledError, match="naive and the vendor side"):
        three_numbers({**point, "sides": {}}, NAMES, TILES, 1.1)


def test_the_section_names_the_vendor_call_and_never_claims_cublas():
    point = synthetic_point(
        {NAIVE_KERNEL: [1000] * 3, TORCH_MATMUL: [100] * 3}, variants=()
    )

    section = comparison([point], NAMES, TILES, 1.1, "2.15.0")

    assert section["vendor"] == (
        "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0"
    )
    assert section["cublas"] == CUBLAS
    assert (section["tiled_rule"], section["ratio_rule"]) == (TILED_RULE, RATIO_RULE)
    assert section["min_faster_ratio"] == 1.1
    assert "cuBLAS" not in section["vendor"] and "kernel" not in section["vendor"]
    with pytest.raises(MatmulTiledError, match="PyTorch version"):
        vendor_label("")
