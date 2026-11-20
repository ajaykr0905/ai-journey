import dataclasses
import json

import pytest

from ai_journey.gpu_reduce_ceiling import (
    DEFINITION,
    BandwidthCeiling,
    load_ceiling,
)
from ai_journey.gpu_reduce_plan import PLAN, TORCH_SUM
from ai_journey.gpu_reduce_source import R0, R1, R2, R5, STEP_KERNELS, GpuReduceError
from ai_journey.gpu_reduce_speedup import (
    RATIO_RULE,
    best_kernel,
    speedup,
    step_ratio,
    vendor_label,
)
from ai_journey.kernel_metal_report import host_environment
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON

CEILING = json.loads(CEILING_JSON)


def write_ceiling(directory, report=None):
    path = directory / "gpu-ceilings.json"
    path.write_bytes(canonical_json_bytes(report or CEILING) + b"\n")
    return path


def side(name, median_ns, spread=0.05, moved=1 << 20):
    """A side entry whose quartiles sit ``spread`` around its median."""

    p25, p75 = round(median_ns * (1 - spread)), round(median_ns * (1 + spread))
    return {
        "implementation": name,
        "timing": {"median_ns": median_ns, "p25_ns": p25, "p75_ns": p75},
        "median_gb_per_second": moved / median_ns,
        "iqr_gb_per_second": [moved / p75, moved / p25],
    }


def point(n, times, index=0):
    return {
        "index": index,
        "n": n,
        "sides": {name: side(name, value) for name, value in times.items()},
    }


# Times in ns: R0 slow, R1 and R2 each faster, then no clear changes.
TIMES = {
    R0: 9000,
    R1: 3000,
    R2: 2000,
    "reduce_r3_simd_shuffle": 2000,
    "reduce_r4_simd_sum": 1980,
    R5: 2010,
    TORCH_SUM: 1500,
}


def test_the_ratio_rule_needs_the_ratio_and_separate_quartiles():
    fast, slow = side("a", 1000), side("b", 2000)
    assert step_ratio(fast, slow, 1.1)["verdict"] == "faster"
    assert step_ratio(slow, fast, 1.1)["verdict"] == "slower"
    near = side("c", 1050)
    assert step_ratio(fast, near, 1.1) == {
        "first": "a",
        "second": "c",
        "ratio": pytest.approx(1.05),
        "iqr_separated": False,
        "verdict": "no_clear_difference",
    }
    wide = side("d", 2000, spread=0.6)
    assert step_ratio(fast, wide, 1.1)["verdict"] == "no_clear_difference"
    assert "GB/s" in RATIO_RULE and "p75" in RATIO_RULE


def test_each_step_gets_its_base_ratio_its_r0_ratio_and_its_verdict():
    section = speedup([point(1 << 24, TIMES)], 1.1, "2.15.0", 50.0)

    entry = section["points"][0]
    rows = {row["step"]: row for row in entry["rows"]}
    assert entry["size_class"] == "large"
    assert [row["kernel"] for row in entry["rows"]] == list(STEP_KERNELS)
    assert rows["R0"]["base"] == TORCH_SUM and rows["R0"]["to_r0"] is None
    assert rows["R0"]["to_base"]["verdict"] == "slower" and rows["R0"]["held"]
    assert rows["R1"]["to_base"]["ratio"] == pytest.approx(3.0)
    assert rows["R1"]["to_base"]["verdict"] == "faster" and rows["R1"]["held"]
    assert rows["R2"]["to_r0"]["ratio"] == pytest.approx(4.5)
    assert [rows[step]["held"] for step in ("R2", "R3", "R4", "R5")] == [True] * 4
    assert rows["R2"]["fraction_of_ceiling"] == pytest.approx((1 << 20) / 2000 / 50)
    assert rows["R2"]["fraction_of_torch_sum"] == pytest.approx(0.75)
    assert entry["torch_sum"]["fraction_of_ceiling"] == pytest.approx(
        (1 << 20) / 1500 / 50
    )
    assert entry["best"]["kernel"] == "reduce_r4_simd_sum"
    assert entry["best"]["to_r0"]["verdict"] == "faster"
    assert section["vendor"] == vendor_label("2.15.0")
    assert section["ceiling_gb_per_second"] == 50.0
    assert section["hypotheses"][1] == {
        "step": "R1",
        "kernel": R1,
        "base": R0,
        "held_at": [0],
        "missed_at": [],
    }


def test_a_prediction_holds_only_for_the_size_class_of_its_point():
    small = point(1 << 16, TIMES, index=0)
    large = point(1 << 26, TIMES, index=1)

    records = speedup([small, large], 1.1, "2.15.0", 50.0)["hypotheses"]

    # At 2^16 the plan predicts no clear difference from R1 to R2.
    assert records[2] == {
        "step": "R2",
        "kernel": R2,
        "base": R1,
        "held_at": [1],
        "missed_at": [0],
    }
    assert PLAN[2].predicted(1 << 16) == ("no_clear_difference",)


def test_the_best_kernel_breaks_a_tie_by_step_order_and_needs_every_side():
    tie = point(1 << 20, {**TIMES, R1: 1000, R2: 1000})
    assert best_kernel(tie) == R1
    fastest = point(1 << 20, {**TIMES, R0: 100})
    assert speedup([fastest], 1.1, "2.15.0", 50.0)["points"][0]["best"]["to_r0"] is None
    partial = point(1 << 20, {name: 1000 for name in STEP_KERNELS})
    with pytest.raises(GpuReduceError, match="every step kernel and torch.sum"):
        speedup([partial], 1.1, "2.15.0", 50.0)
    with pytest.raises(GpuReduceError, match="PyTorch version"):
        vendor_label("")


def test_the_ceiling_comes_only_from_a_verified_gpu_ceilings_report(tmp_path):
    ceiling = load_ceiling(write_ceiling(tmp_path))
    summary = CEILING["summary"]

    assert ceiling.report_sha256 == CEILING["report_sha256"]
    assert ceiling.gb_per_second == summary["bandwidth_ceiling_gb_per_second"]
    assert ceiling.kernel == summary["bandwidth_ceiling_kernel"]
    assert ceiling.buffer_bytes == summary["bandwidth_ceiling_buffer_bytes"]
    assert ceiling.measured is True
    assert ceiling.load_average_before == summary["load_average_before"]
    assert ceiling.to_mapping()["definition"] == DEFINITION
    assert ceiling.same_environment(CEILING["environment"])
    other = {**CEILING["environment"], "torch_version": "2.14.0"}
    assert not ceiling.same_environment(other)
    assert ceiling.same_environment(host_environment("2.15.0")) == (
        host_environment("2.15.0") == CEILING["environment"]
    )
    tampered = {
        **CEILING,
        "summary": {**summary, "bandwidth_ceiling_gb_per_second": 1e3},
    }
    with pytest.raises(GpuReduceError, match="ceiling report was refused"):
        load_ceiling(write_ceiling(tmp_path, tampered))
    with pytest.raises(GpuReduceError, match="label must match"):
        dataclasses.replace(ceiling, measured=False)
    with pytest.raises(GpuReduceError, match="stream kernel"):
        dataclasses.replace(ceiling, kernel="fma_chains_8")
    with pytest.raises(GpuReduceError, match="positive float"):
        BandwidthCeiling(**{**vars(ceiling), "gb_per_second": 0.0})
