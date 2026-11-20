import pytest

from ai_journey.kernel_metal_benchmark import DeviceTiming
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_speedup import best_kernel, speedup
from ai_journey.matmul_checklist_steps import CHECKLIST
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_benchmark import _rates
from ai_journey.matmul_metal_runtime import TORCH_MATMUL

FLOPS = 2 * 64 * 64 * 64
# Median sample times in ns; a smaller time is a higher rate.
TIMES = {
    "matmul_x_columns": 1000,
    "matmul_x_rows": 2500,
    "matmul_tiled_32": 1100,
    "matmul_s2_coarse_4x1": 600,
    "matmul_s3_block_4x4": 300,
    "matmul_s4_vec4": 290,
    "matmul_s5_pad4": 300,
    "matmul_s6_group_8x8": 500,
    "matmul_s6_group_32x32": 310,
    "matmul_s7_interior": 250,
    TORCH_MATMUL: 50,
}


def side(name, median, spread=10):
    samples = (median - spread, median, median + spread)
    return _rates(DeviceTiming(name, 1, 1, 3, samples, "injected"), FLOPS)


def point(times=TIMES, index=0):
    return {
        "index": index,
        "m": 64,
        "n": 64,
        "k": 64,
        "sides": {name: side(name, median) for name, median in times.items()},
    }


def rows_by_variant(section, index=0):
    return {row["variant"]: row for row in section["points"][index]["rows"]}


def test_each_change_has_its_ratio_to_its_base_to_s0_and_to_torch():
    section = speedup([point()], 1.1, "2.15.0")
    rows = rows_by_variant(section)

    assert section["vendor"] == "vendor GPU matmul: torch.matmul on MPS, PyTorch 2.15.0"
    assert section["cublas"] == "not measured, no NVIDIA GPU or cuBLAS on this host"
    assert [row["variant"] for row in section["points"][0]["rows"]] == [
        name for step in CHECKLIST for name in step.variants
    ]
    s0 = rows["matmul_x_columns"]
    assert s0["base"] == "matmul_x_rows"
    assert s0["to_base"]["ratio"] == pytest.approx(2.5)
    assert s0["to_base"]["verdict"] == "faster" and s0["held"] is True
    assert s0["to_s0"]["ratio"] == pytest.approx(1.0)
    assert s0["to_s0"]["verdict"] == "no_clear_difference"
    assert s0["base_rate"]["median_gflops_per_second"] == FLOPS / 2500
    s2 = rows["matmul_s2_coarse_4x1"]
    assert s2["to_base"]["ratio"] == pytest.approx(1100 / 600)
    assert s2["to_s0"]["ratio"] == pytest.approx(1000 / 600)
    assert s2["fraction_of_torch"] == pytest.approx(50 / 600)
    assert rows["matmul_s4_vec4"]["to_base"]["verdict"] == "no_clear_difference"
    assert rows["matmul_s4_vec4"]["held"] is True
    assert rows["matmul_s6_group_8x8"]["to_base"]["verdict"] == "slower"
    assert rows["matmul_s6_group_32x32"]["held"] is True
    assert rows["matmul_s7_interior"]["to_base"]["verdict"] == "faster"
    assert rows["matmul_tiled_32"]["held"] is True


def test_a_prediction_that_the_run_contradicts_is_recorded_as_missed():
    times = {**TIMES, "matmul_s5_pad4": 200, "matmul_s3_block_4x4": 640}
    section = speedup([point(times), point(index=1)], 1.1, "2.15.0")

    records = {record["variant"]: record for record in section["hypotheses"]}
    assert records["matmul_s5_pad4"] == {
        "step": "S5",
        "variant": "matmul_s5_pad4",
        "predicted": ["no_clear_difference", "slower"],
        "held_at": [1],
        "missed_at": [0],
        "no_verdict_at": [],
    }
    assert records["matmul_s3_block_4x4"]["missed_at"] == [0]
    assert rows_by_variant(section)["matmul_s3_block_4x4"]["to_base"]["verdict"] == (
        "no_clear_difference"
    )


def test_a_refused_kernel_has_no_numbers_and_its_dependants_no_verdict():
    times = {name: time for name, time in TIMES.items() if name != "matmul_s5_pad4"}
    section = speedup([point(times)], 1.1, "2.15.0")
    rows = rows_by_variant(section)

    pad = rows["matmul_s5_pad4"]
    assert (pad["ran"], pad["rate"], pad["to_base"], pad["to_s0"]) == (
        False,
        None,
        None,
        None,
    )
    assert pad["fraction_of_torch"] is None and pad["held"] is None
    for name in ("matmul_s6_group_8x8", "matmul_s6_group_32x32", "matmul_s7_interior"):
        assert rows[name]["ran"] is True and rows[name]["base_rate"] is None
        assert rows[name]["to_base"] is None and rows[name]["held"] is None
        assert rows[name]["to_s0"] is not None
    records = {record["variant"]: record for record in section["hypotheses"]}
    assert records["matmul_s7_interior"]["no_verdict_at"] == [0]


def test_the_best_kernel_is_the_fastest_timed_one_and_a_tie_keeps_the_first():
    assert best_kernel(point()) == "matmul_s7_interior"
    tied = {**TIMES, "matmul_s7_interior": 290}
    assert best_kernel(point(tied)) == "matmul_s4_vec4"
    best = speedup([point()], 1.1, "2.15.0")["points"][0]["best"]
    assert best["variant"] == "matmul_s7_interior"
    assert best["to_s0"]["ratio"] == pytest.approx(4.0)
    assert best["fraction_of_torch"] == pytest.approx(0.2)
    assert set(VARIANTS) | {TORCH_MATMUL} == set(TIMES)


def test_a_point_without_s0_or_torch_is_refused():
    for missing in ("matmul_x_columns", TORCH_MATMUL):
        times = {name: time for name, time in TIMES.items() if name != missing}
        with pytest.raises(MatmulChecklistError, match="S0 kernel and torch"):
            speedup([point(times)], 1.1, "2.15.0")
