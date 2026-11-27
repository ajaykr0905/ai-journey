"""Tests of the explained speedup model, a model of declared inputs only.

No number here is a measurement. The expected values follow from the rule
by hand: 2 N^3 FLOPs over the listed dense peak, N^2 (2 b + 4) bytes over
the listed bandwidth, and the larger of the two.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_tc_config import PATHS, load_config
from ai_journey.triton_tc_kernel import TritonTcError
from ai_journey.triton_tc_model import (
    LABEL,
    RULE,
    _gflops_text,
    model_mapping,
    point,
    ratio_text,
    run_model,
    traffic_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "day-82-triton-tc.json")
MODEL = run_model(CONFIG)
A100, H100 = MODEL["devices"]


def row(device, size):
    (found,) = [item for item in device["rows"] if item["size"] == size]
    return found


def test_the_model_states_its_rule_and_its_label():
    assert MODEL["label"] == LABEL == "declared_model_not_a_measurement"
    assert MODEL["rule"] == list(RULE)
    assert "time = max(flops / compute ceiling, bytes / bandwidth)" in RULE
    assert MODEL["sizes"] == [512, 1024, 2048, 4096, 8192]
    assert [device["target"] for device in MODEL["devices"]] == ["sm_80", "sm_90"]


def test_one_point_follows_the_rule_by_hand():
    device = CONFIG.device("sm_80")
    fp32 = point(device, 512, "fp32")
    bf16 = point(device, 512, "bf16")

    assert fp32["flops"] == 2 * 512**3 == 268_435_456
    assert fp32["bytes"] == 12 * 512**2
    assert fp32["compute_us"] == Fraction(268_435_456, 19_500_000)
    assert fp32["memory_us"] == Fraction(12 * 512**2, 2_039_000)
    assert (fp32["bound"], fp32["time_us"]) == ("compute", fp32["compute_us"])
    assert bf16["bytes"] == 8 * 512**2
    assert bf16["compute_us"] == Fraction(268_435_456, 312_000_000)
    assert (bf16["bound"], bf16["time_us"]) == ("memory", bf16["memory_us"])
    assert bf16["ridge"] == Fraction(312_000, 2039)
    assert traffic_bytes(4096, "tf32") == 12 * 4096**2
    assert traffic_bytes(4096, "fp16") == 8 * 4096**2


def test_bf16_does_16_and_tf32_8_times_the_fp32_fmas_per_clock_per_sm():
    for device in (A100, H100):
        assert device["per_clock_ratio"] == {
            "fp32": 16,
            "tf32": 2,
            "fp16": 1,
            "bf16": 1,
        }
    assert A100["peak_ratio"] == {"fp32": 16, "tf32": 2, "fp16": 1, "bf16": 1}
    assert H100["peak_ratio"]["fp32"] == Fraction("989.5") / 67
    assert H100["peak_ratio"]["tf32"] == Fraction("989.5") / Fraction("494.5")


def test_the_memory_bound_sizes_are_the_small_tensor_core_sizes():
    bound = {
        (device["key"], item["size"]): item["memory_bound"]
        for device in (A100, H100)
        for item in device["rows"]
        if item["memory_bound"]
    }
    assert bound == {
        ("a100-sxm4-80gb", 512): ["fp16", "bf16"],
        ("h100-sxm5-80gb", 512): ["tf32", "fp16", "bf16"],
        ("h100-sxm5-80gb", 1024): ["fp16", "bf16"],
    }


def test_a_compute_bound_size_has_the_peak_ratio_and_a_memory_bound_one_less():
    for device in (A100, H100):
        for item in device["rows"]:
            if not item["memory_bound"]:
                assert item["ratio"] == device["peak_ratio"], item["size"]
            else:
                assert item["ratio"]["fp32"] < device["peak_ratio"]["fp32"]
    assert ratio_text(row(A100, 512)["ratio"]) == "13.38 : 1.67 : 1"
    assert ratio_text(row(A100, 8192)["ratio"]) == "16.00 : 2.00 : 1"
    assert ratio_text(row(H100, 512)["ratio"]) == "6.40 : 1.50 : 1"
    assert ratio_text(row(H100, 1024)["ratio"]) == "12.80 : 1.73 : 1"
    assert ratio_text(row(H100, 2048)["ratio"]) == "14.77 : 2.00 : 1"


def test_the_fp16_path_has_the_bf16_time_in_this_model():
    for device in (A100, H100):
        for item in device["rows"]:
            assert item["ratio"]["fp16"] == 1
            assert item["points"]["fp16"] == {
                **item["points"]["bf16"],
                "path": "fp16",
            }


def test_the_model_mapping_is_exact_canonical_json():
    mapping = model_mapping(CONFIG)
    encoded = canonical_json_bytes(mapping)
    point_512 = mapping["devices"][0]["rows"][0]["points"]["fp32"]

    assert len(encoded) < 64 * 1024
    assert point_512["time_us"] == {
        # 268435456 / 19500000 microseconds, reduced by 32.
        "numerator": 8388608,
        "denominator": 609375,
        "decimal": "13.765921",
    }
    assert mapping["devices"][1]["implied_clock_mhz"]["fp32"]["decimal"] == (
        "1982.717803"
    )
    assert set(mapping["devices"][0]["rows"][0]["ratio"]) == set(PATHS)


def test_the_model_refuses_inputs_outside_its_rule():
    with pytest.raises(TritonTcError, match="path must be one of"):
        traffic_bytes(512, "fp8")
    with pytest.raises(TritonTcError, match="positive integer"):
        traffic_bytes(0, "bf16")
    with pytest.raises(TypeError, match="TcModelConfig"):
        run_model({})
    with pytest.raises(TritonTcError, match="more than 6 decimals"):
        _gflops_text(Fraction(1, 3))
    assert _gflops_text(Fraction(989_500)) == "989500"
    assert _gflops_text(Fraction("0.25")) == "0.25"
