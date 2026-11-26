import pytest

from ai_journey.metal_tc_benchmark import benchmark_point, compulsory_bytes
from ai_journey.metal_tc_ceiling import load_tc_ceiling, ridge_point, roofline
from ai_journey.metal_tc_config import TcPoint
from ai_journey.metal_tc_plan import PLAN, SIDES, TORCH_BFLOAT16, TORCH_FLOAT16
from ai_journey.metal_tc_source import M1, M2, M3, S1, MetalTcError
from ai_journey.metal_tc_speedup import speedup
from test_metal_fused_ceiling import write_recorded
from test_metal_liger_emulator import ticking_clock
from test_metal_tc_emulator import TcEmulator

POINT = TcPoint(1, 40)


def run(backend=None, clock=None, repeats=7):
    return benchmark_point(
        backend or TcEmulator(),
        seed=82,
        point=POINT,
        check_rows=6,
        warmup=1,
        repeats=repeats,
        target_flops_per_sample=1 << 20,
        clock=clock or ticking_clock(),
    )


def test_the_compulsory_bytes_follow_the_input_and_output_types():
    assert compulsory_bytes(S1, 512) == compulsory_bytes(M1, 512) == 12 * 512**2
    assert compulsory_bytes(M2, 512) == compulsory_bytes(M3, 512) == 8 * 512**2
    assert compulsory_bytes(TORCH_FLOAT16, 512) == 6 * 512**2
    assert compulsory_bytes("torch_float32", 512) == 12 * 512**2
    with pytest.raises(MetalTcError):
        compulsory_bytes("cublas", 512)


def test_a_point_times_seven_sides_with_one_batch_and_checks_them():
    mapping = run().to_mapping()

    assert mapping["side_order"] == list(SIDES)
    assert mapping["supported"] == list(SIDES)
    assert mapping["size"] == 40 and mapping["launch"]["groups"] == [4]
    calls = {entry["calls_per_sample"] for entry in mapping["sides"].values()}
    assert calls == {-(-(1 << 20) // (2 * 40**3))}
    for side in SIDES:
        entry = mapping["sides"][side]
        assert len(entry["samples_ns"]) == 7
        assert entry["timing"]["median_ns"] > 0
        assert (
            entry["intensity_flops_per_byte"]
            == 2 * 40**3 / entry["compulsory_bytes_per_call"]
        )
        assert mapping["checks"][side]["max_bound_ratio"] <= 1
    assert mapping["m1_same_bits_as_s1"] is True


def test_an_unsupported_side_keeps_its_turn_but_records_no_samples():
    mapping = run(TcEmulator(bfloat=False, torch_bfloat=False)).to_mapping()

    assert mapping["side_order"] == list(SIDES)
    assert M3 not in mapping["sides"] and TORCH_BFLOAT16 not in mapping["sides"]
    assert set(mapping["checks"]) == set(mapping["sides"])


def test_a_point_whose_output_drifts_stops_the_run():
    def drift(side, output):
        if side == M2:
            output = output.copy()
            output[0] += 1.0
        return output

    with pytest.raises(MetalTcError, match="exceeds its bound"):
        run(TcEmulator(corrupt=drift))


def _side(name, samples, size=64):
    flops = 2 * size**3
    compulsory = 12 * size**2
    ordered = sorted(samples)
    return {
        "implementation": name,
        "timing": {
            "p25_ns": ordered[0],
            "median_ns": ordered[1],
            "p75_ns": ordered[2],
        },
        "tflops_per_second": flops / ordered[1] / 1000,
        "iqr_tflops_per_second": [flops / ordered[2] / 1000, flops / ordered[0] / 1000],
        "median_ns_per_call": float(ordered[1]),
        "intensity_flops_per_byte": flops / compulsory,
    }


def test_each_step_gets_the_declared_ratio_rule_and_its_record():
    times = {
        "s1": [300, 310, 320],
        "m1": [200, 205, 210],
        "m2": [199, 204, 211],
        "torch_float32": [100, 101, 102],
        "torch_float16": [99, 100, 101],
    }
    point = {
        "index": 0,
        "size": 64,
        "sides": {n: _side(n, t) for n, t in times.items()},
    }
    gains = speedup([point], 100.0, 4000.0, 1.1, "2.15.0")
    steps = {step["step"]: step for step in gains["points"][0]["steps"]}

    assert steps["m1_over_s1"]["verdict"] == "faster"
    assert steps["m1_over_s1"]["ratio"] == 310 / 205
    assert steps["m2_over_m1"]["verdict"] == "no_clear_difference"
    assert steps["m1_over_torch_float32"]["verdict"] == "slower"
    assert steps["m3_over_m1"]["ran"] is False and steps["m3_over_m1"]["held"] is None
    records = {record["step"]: record for record in gains["hypotheses"]}
    assert records["m1_over_s1"]["held_at"] == [0]
    assert records["m3_over_s1"]["not_run_at"] == [0]
    assert len(records) == len(PLAN)
    rate = gains["points"][0]["rates"]["s1"]
    intensity = 2 * 64**3 / (12 * 64**2)
    assert rate["roofline_gflops_per_second"] == min(4000.0, intensity * 100.0)
    assert gains["ridge_flops_per_byte"] == 40.0


def test_the_roofline_takes_the_lower_ceiling(tmp_path):
    assert roofline(100.0, 100.0, 4000.0) == 4000.0
    assert roofline(10.0, 100.0, 4000.0) == 1000.0
    with pytest.raises(MetalTcError):
        roofline(10, 100.0, 4000.0)
    ceiling = load_tc_ceiling(write_recorded(tmp_path))
    assert ridge_point(ceiling) == (
        ceiling.fma_gflops_per_second / ceiling.bandwidth.gb_per_second
    )


def test_a_tampered_ceiling_report_sets_no_ceiling(tmp_path):
    path = write_recorded(tmp_path)
    path.write_text(path.read_text().replace('"27.2"', '"27.3"'), encoding="ascii")

    with pytest.raises(MetalTcError, match="refused"):
        load_tc_ceiling(path)
