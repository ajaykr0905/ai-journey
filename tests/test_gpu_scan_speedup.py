import pytest

from ai_journey.gpu_scan_benchmark import ScanPoint, modeled_bytes
from ai_journey.gpu_scan_plan import COPY, VENDOR
from ai_journey.gpu_scan_runtime import copy_launch, scan_plan
from ai_journey.gpu_scan_source import (
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
)
from ai_journey.gpu_scan_speedup import (
    RATIO_RULE,
    best_kernel,
    ratio_step,
    speedup,
    vendor_label,
)
from ai_journey.kernel_metal_benchmark import DeviceTiming
from test_gpu_scan_correctness import StandInBackend

LIMITS = StandInBackend().limits


def make_point(length, samples, index=0):
    """A point mapping whose sides took the given sample times, two calls each."""

    names = [name for name in KERNELS if name in samples]
    timings = tuple(
        DeviceTiming(name, 2, 1, len(samples[name]), tuple(samples[name]), "injected")
        for name in (*names, VENDOR, COPY)
    )
    return ScanPoint(
        index,
        length,
        {name: scan_plan(name, LIMITS, length) for name in names},
        copy_launch(LIMITS, length),
        timings,
        {},
    ).to_mapping()


def rows(gains, point=0):
    return {row["kernel"]: row for row in gains["points"][point]["rows"]}


def test_each_kernel_gets_its_ratios_verdicts_and_copy_fractions():
    point = make_point(
        65536,
        {
            S0_HILLIS_STEELE: [1000, 1000, 1000],
            S1_BLELLOCH: [1040, 1050, 1060],
            S2_SIMD: [500, 510, 520],
            VENDOR: [2000, 2010, 2020],
            COPY: [490, 500, 510],
        },
    )

    gains = speedup([point], 1.1, "2.15.0")
    table = rows(gains)

    assert gains["vendor"] == "vendor GPU scan: torch.cumsum on MPS, PyTorch 2.15.0"
    assert gains["ratio_rule"] == RATIO_RULE and gains["min_faster_ratio"] == 1.1
    s0, s1, s2 = table[S0_HILLIS_STEELE], table[S1_BLELLOCH], table[S2_SIMD]
    assert s0["base"] == VENDOR and s0["to_s0"] is None
    assert s0["to_base"]["ratio"] == pytest.approx(2.01)
    assert s0["to_base"]["verdict"] == "faster"
    assert s0["predicted"] == ["slower"] and s0["held"] is False
    assert s1["to_base"]["verdict"] == "no_clear_difference" and s1["held"] is True
    assert s2["to_base"]["verdict"] == "faster" and s2["held"] is False
    assert s2["to_s0"] == s2["to_base"]
    assert s2["to_vendor"]["ratio"] == pytest.approx(2010 / 510)
    moved = 2 * 8 * 65536
    copy_rate = 2 * 32 * 16384 / 500
    assert s0["rate"] == {
        "median_gb_per_second": moved / 1000,
        "iqr_gb_per_second": [moved / 1000, moved / 1000],
    }
    assert s0["fraction_of_copy"] == pytest.approx((moved / 1000) / copy_rate)
    modeled = 2 * modeled_bytes(65536) / 1000
    assert s0["modeled_gb_per_second"] == modeled
    assert s0["modeled_fraction_of_copy"] == pytest.approx(modeled / copy_rate)
    entry = gains["points"][0]
    assert entry["best"]["kernel"] == S2_SIMD
    assert entry["best"]["to_vendor"]["verdict"] == "faster"
    assert entry["vendor"]["fraction_of_copy"] == pytest.approx(
        (moved / 2010) / copy_rate
    )


def test_overlapping_samples_or_a_small_ratio_give_no_clear_difference():
    first = {"implementation": "a", "median_gb_per_second": 12.0}
    second = {"implementation": "b", "median_gb_per_second": 10.0}
    first["timing"] = {"p25_ns": 80, "p75_ns": 95}
    second["timing"] = {"p25_ns": 90, "p75_ns": 110}

    assert ratio_step(first, second, 1.1)["verdict"] == "no_clear_difference"
    first["timing"] = {"p25_ns": 70, "p75_ns": 85}
    assert ratio_step(first, second, 1.1)["verdict"] == "faster"
    assert ratio_step(second, first, 1.1)["verdict"] == "slower"
    assert ratio_step(first, second, 1.3)["verdict"] == "no_clear_difference"
    assert ratio_step(first, second, 1.3)["iqr_separated"] is True


def test_a_length_without_a_prediction_and_a_refused_kernel_give_no_verdict():
    pinned = make_point(
        1 << 20,
        {
            S0_HILLIS_STEELE: [1000, 1000, 1000],
            S1_BLELLOCH: [1500, 1500, 1500],
            VENDOR: [900, 900, 900],
            COPY: [400, 400, 400],
        },
    )
    other = make_point(
        4096,
        {
            S0_HILLIS_STEELE: [1000, 1000, 1000],
            S1_BLELLOCH: [1000, 1000, 1000],
            VENDOR: [900, 900, 900],
            COPY: [400, 400, 400],
        },
        index=1,
    )

    gains = speedup([pinned, other], 1.1, "2.15.0")

    assert rows(gains)[S1_BLELLOCH]["to_base"]["verdict"] == "slower"
    assert rows(gains)[S1_BLELLOCH]["held"] is True
    missing = rows(gains)[S2_SIMD]
    assert (missing["ran"], missing["rate"], missing["to_base"], missing["held"]) == (
        False,
        None,
        None,
        None,
    )
    assert rows(gains, 1)[S1_BLELLOCH]["predicted"] is None
    records = {record["kernel"]: record for record in gains["hypotheses"]}
    assert records[S1_BLELLOCH] == {
        "step": "S1",
        "kernel": S1_BLELLOCH,
        "base": S0_HILLIS_STEELE,
        "held_at": [0],
        "missed_at": [],
        "no_verdict_at": [],
        "no_prediction_at": [1],
    }
    assert records[S2_SIMD]["no_verdict_at"] == [0, 1]
    assert records[S0_HILLIS_STEELE]["held_at"] == [0]
    assert best_kernel(pinned) == S0_HILLIS_STEELE


def test_the_vendor_label_names_the_call_and_a_point_needs_vendor_and_copy():
    assert vendor_label("2.15.0").startswith("vendor GPU scan: torch.cumsum on MPS")
    with pytest.raises(GpuScanError, match="PyTorch version"):
        vendor_label("")
    point = make_point(
        4096, {name: [1000, 1000, 1000] for name in (S0_HILLIS_STEELE, VENDOR, COPY)}
    )
    del point["sides"][COPY]
    with pytest.raises(GpuScanError, match="needs torch.cumsum and the copy"):
        speedup([point], 1.1, "2.15.0")
