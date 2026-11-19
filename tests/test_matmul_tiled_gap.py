import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_ceiling_roofline import MeasuredRoofline
from ai_journey.kernel_metal_report import SYNTHETIC_LABEL
from ai_journey.matmul_bottleneck_config import config_from_mapping as rule_from_mapping
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_bottleneck_rule import PointInputs, classify
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_config import config_from_mapping
from ai_journey.matmul_tiled_gap import (
    CEILING_SOURCE,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    TILED_SOURCE,
    TIMING_SOURCE,
    build_gap_report,
)
from ai_journey.matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
    measure_timing_report,
)
from ai_journey.matmul_tiled_gap_verify import verify_gap_report
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL, TiledWorkload
from ai_journey.matmul_tiled_variants import VARIANTS
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from ai_journey.memory_access_tiling import (
    MatmulShape,
    NaiveMatmulTraffic,
    TiledMatmulTraffic,
)
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_matmul_tiled_correctness import StandInBackend
from test_matmul_tiled_gap_timing import LOAD, TimingStandIn
from test_matmul_tiled_recorded import RECORDED_CONFIG
from test_matmul_tiled_recorded import RECORDED_REPORT_JSON as TILED_JSON

ROOT = Path(__file__).resolve().parents[1]
RULE = load_rule_config(ROOT / "config/day-75-matmul-bottleneck.json")
SIDES = [NAIVE_KERNEL, *VARIANTS, TORCH_MATMUL]
FLOOR_NS = 20_000


def tiled():
    return json.loads(TILED_JSON)


def ceilings():
    return json.loads(CEILING_JSON)


def steady_costs(report, scale=None):
    """The steady time per call of each side at each point, scaled per side."""

    scale = scale or {}
    costs = {}
    for point in report["benchmark"]["points"]:
        shape = (point["m"], point["n"], point["k"])
        for side, entry in point["sides"].items():
            per_call = entry["timing"]["median_ns"] // entry["calls_per_sample"]
            costs[shape, side] = per_call * scale.get(side, 1)
    return costs


class VirtualClock:
    """A clock that only the stand-in's calls move, one nanosecond per reading."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1
        return self.now


class ClockEvents:
    """Two events that read the virtual clock, as MPS events bracket a call."""

    def __init__(self, clock):
        self.clock = clock
        self.started = self.stopped = 0

    def start(self):
        self.started = self.clock.now

    def stop(self):
        self.stopped = self.clock.now

    def elapsed_ms(self):
        return (self.stopped - self.started) / 1e6


class PacedStandIn(TimingStandIn):
    """Each call moves the virtual clock by its side's cost at that shape."""

    def __init__(self, clock, costs, **kwargs):
        super().__init__(**kwargs)
        self.clock, self.costs = clock, costs

    def event_pair(self):
        return ClockEvents(self.clock)

    def workload(self, left, right, launches):
        bound = StandInBackend.workload(self, left, right, launches)
        shape = (left.shape[0], right.shape[1], left.shape[1])

        def paced(name):
            def call():
                self.clock.now += self.costs.get((shape, name), FLOOR_NS)

            return call

        return TiledWorkload(
            sides={name: paced(name) for name in bound.sides},
            synchronize=bound.synchronize,
            check=bound.check,
        )


def timing_report(scale=None, environment=None, mapping=None, **backend_options):
    """A synthetic timing of the recorded tiled config, on the recorded host.

    The stand-in reads the environment of the test host, so the report is
    pinned to the recorded host's tokens and sealed again. It stays
    synthetic, as its backend and timer say.
    """

    clock = VirtualClock()
    backend = PacedStandIn(clock, steady_costs(tiled(), scale), **backend_options)
    report = measure_timing_report(
        config_from_mapping(mapping or RECORDED_CONFIG),
        backend=backend,
        clock=clock,
        load_average=lambda: LOAD,
    )
    del report[DIGEST_FIELD]
    report["environment"] = dict(environment or tiled()["environment"])
    return seal_report(report)


def gap(timing=None, rule=RULE):
    return build_gap_report(tiled(), ceilings(), timing or timing_report(), rule)


def resealed(report, mutate):
    changed = copy.deepcopy(report)
    del changed[DIGEST_FIELD]
    mutate(changed)
    return seal_report(changed)


@pytest.fixture(scope="module")
def report():
    return gap(timing_report(scale={NAIVE_KERNEL: 3}))


def test_a_gap_joins_three_verified_sources_and_verifies_alone(report):
    summary = verify_gap_report(json.loads(json.dumps(report)), config=RULE)

    assert (report["measured"], report["evidence_boundary"]) == (
        False,
        SYNTHETIC_BOUNDARY,
    )
    assert report["sources"][TILED_SOURCE]["report_sha256"] == tiled()[DIGEST_FIELD]
    assert report["sources"][TILED_SOURCE]["measured"] is True
    assert (
        report["sources"][CEILING_SOURCE]["report_sha256"] == (ceilings()[DIGEST_FIELD])
    )
    assert report["sources"][TIMING_SOURCE]["measured"] is False
    assert (
        report["sources"][CEILING_SOURCE]["load_average_before"]
        == (ceilings()["host_load"]["before"])
    )
    assert report["sides"] == SIDES == summary["sides"]
    assert [[point["m"], point["n"], point["k"]] for point in report["points"]] == [
        [256, 256, 256],
        [200, 300, 100],
    ]
    assert report["rule"]["config_sha256"] == RULE.fingerprint()
    assert report["fastest_calls"][NAIVE_KERNEL]["m"] == 1
    assert summary["sources_checked"] is False
    assert MEASUREMENT_BOUNDARY["inference_from_timings"] is True
    assert MEASUREMENT_BOUNDARY["hardware_counters_used"] is False


def test_each_kernel_side_carries_its_traffic_model_and_attainable_rate(report):
    fma = report["ceilings"]["fma_gflops_per_second"]
    bandwidth = report["ceilings"]["bandwidth_gb_per_second"]
    roofline = MeasuredRoofline(fma, bandwidth, "0" * 64, False, SYNTHETIC_LABEL)
    point = report["points"][0]
    shape = MatmulShape(256, 256, 256)

    for side in SIDES:
        entry = point["sides"][side]
        traffic = entry["traffic"]
        if side == TORCH_MATMUL:
            assert traffic is None
            continue
        model = (
            NaiveMatmulTraffic(shape)
            if side == NAIVE_KERNEL
            else TiledMatmulTraffic(shape, VARIANTS[side].tile)
        )
        assert traffic["bytes"] == model.total_bytes
        expected = roofline.below_ceiling(
            entry["median_gflops_per_second"], model.arithmetic_intensity
        )
        assert (
            traffic["attainable_gflops_per_second"]
            == (expected["attainable_gflops_per_second"])
        )
        assert traffic["attainable_share"] == expected["fraction_of_ceiling"]
        assert entry["fma_ceiling_share"] == entry["median_gflops_per_second"] / fma
    naive = point["sides"][NAIVE_KERNEL]["traffic"]
    assert (naive["model"], naive["tile"], naive["bytes"]) == (
        "no_reuse",
        None,
        4 * (2 * 256**3 + 256 * 256),
    )
    # The naive kernel beat the bandwidth roof of its no-reuse traffic.
    assert naive["attainable_share"] > 1
    assert report["summary"]["points_above_modeled_roof"][NAIVE_KERNEL] == 2
    assert point["ideal_reuse"]["bytes"] == 4 * 3 * 256 * 256


def test_each_name_is_the_day_75_rule_with_the_ideal_reuse_roof(report):
    fma = report["ceilings"]["fma_gflops_per_second"]

    for point in report["points"]:
        roof = point["ideal_reuse"]["bandwidth_roof_gflops_per_second"]
        for side in SIDES:
            entry = point["sides"][side]
            one_call = entry["one_call"]
            inputs = PointInputs(
                steady_ns_per_call=entry["steady_ns_per_call"],
                rate_gflops_per_second=entry["median_gflops_per_second"],
                gpu_ns=one_call["median_gpu_ns"],
                host_ns=one_call["median_host_ns"],
                dispatch_ns=one_call["median_dispatch_ns"],
                fma_ceiling_gflops_per_second=fma,
                fma_scales_with_work=report["ceilings"]["fma_scales_with_work"],
                bandwidth_roof_gflops_per_second=roof,
                fixed_cost_bound_ns=report["fastest_calls"][side]["median_host_ns"],
            )
            assert entry["decision"] == classify(inputs, RULE.rule).to_mapping()
    # One call of the naive side took three steady times, the others one.
    tops = {side: report["summary"]["top_bottleneck"][side]["top"] for side in SIDES}
    assert tops == {
        NAIVE_KERNEL: "dispatch-bound",
        **{side: "latency or occupancy-bound" for side in SIDES[1:]},
    }


def test_a_rule_config_with_another_roof_is_refused():
    mapping = RULE.to_mapping()
    mapping["rule"]["intensity_model"] = "no_reuse"

    with pytest.raises(MatmulTiledGapError, match="ideal_reuse roof"):
        gap(rule=rule_from_mapping(mapping))


def other_switch(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_FAST_MATH"] = "1"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "environment": {
                    **json.loads(TILED_JSON)["environment"],
                    "torch_version": "2.14",
                }
            },
            "different software",
        ),
        (
            {
                "environment": {
                    **json.loads(TILED_JSON)["environment"],
                    "torch_float32_matmul_precision": "high",
                }
            },
            "precisions",
        ),
        ({"mapping": {**RECORDED_CONFIG, "seed": 77}}, "tiled report's config"),
        ({"threads": {"matmul_tiled_32": 512}}, "admitted other tiles"),
    ],
)
def test_sources_that_do_not_pair_are_refused(options, message):
    with pytest.raises(MatmulTiledGapError, match=message):
        gap(timing_report(**options))


def test_sources_with_other_mps_switches_are_refused():
    timing = resealed(timing_report(), other_switch)

    with pytest.raises(MatmulTiledGapError, match="MPS switches"):
        gap(timing)


def test_a_tampered_source_is_rejected_by_its_own_verifier():
    timing = timing_report()
    tampered_tiled = {**tiled(), "label": "synthetic timing, not a measurement"}
    tampered_ceiling = {**ceilings(), "measured": False}
    tampered_timing = {**timing, "measured": True}

    for arguments, name in (
        ((tampered_tiled, ceilings(), timing), "tiled report"),
        ((tiled(), tampered_ceiling, timing), "gpu-ceilings report"),
        ((tiled(), ceilings(), tampered_timing), "one-call timing report"),
    ):
        with pytest.raises(MatmulTiledGapVerificationError, match=name):
            build_gap_report(*arguments, RULE)


def set_median(report):
    report["points"][0]["sides"]["matmul_tiled_8"]["median_sample_ns"] += 1


def set_host(report):
    report["points"][1]["sides"][TORCH_MATMUL]["one_call"]["median_host_ns"] += 7


def set_fastest(report):
    report["fastest_calls"]["matmul_tiled_16"]["median_host_ns"] = 10**9


def set_fma(report):
    report["ceilings"]["fma_gflops_per_second"] *= 2


def set_ridge(report):
    report["ceilings"]["ridge_flops_per_byte"] = 1.0


def set_threshold(report):
    report["rule"]["config"]["rule"]["near_roof_share_min"] = 0.01


def set_rule_digest(report):
    report["rule"]["config_sha256"] = "0" * 64


def set_bytes(report):
    report["points"][0]["sides"]["matmul_tiled_32"]["traffic"]["bytes"] += 4


def set_share(report):
    report["points"][0]["sides"][NAIVE_KERNEL]["traffic"]["attainable_share"] = 0.5


def set_name(report):
    decision = report["points"][1]["sides"]["matmul_tiled_16"]["decision"]
    decision["bottleneck"] = "memory-bound"


def set_top(report):
    report["summary"]["top_bottleneck"][NAIVE_KERNEL]["top"] = "compute-bound"


def set_sides(report):
    report["sides"] = [NAIVE_KERNEL, "matmul_tiled_8", TORCH_MATMUL]


def set_measured(report):
    report["measured"] = True


def set_source_measured(report):
    report["sources"][TIMING_SOURCE]["measured"] = True


def set_boundary(report):
    report["evidence_boundary"]["hardware_counters_used"] = True


def set_definition(report):
    report["definitions"]["decision"] = "a name from hardware counters"


def set_reason(report):
    report["reasons"]["below_both_roofs"]["bottleneck"] = "memory-bound"


def set_environment(report):
    report["environment"]["numpy_version"] = "numpy 1.26"


def set_switch(report):
    report["torch_mps_switches"]["PYTORCH_MPS_PREFER_METAL"] = "1 1"


def set_configs(report):
    report["sources"][TIMING_SOURCE]["config_sha256"] = "f" * 64


def drop_point(report):
    report["points"] = []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_median, "point 0 does not recompute"),
        (set_host, "point 1 does not recompute"),
        (set_fastest, "faster than the fastest call"),
        (set_fma, "ceilings section does not recompute"),
        (set_ridge, "ceilings section does not recompute"),
        (set_threshold, "rule config digest"),
        (set_rule_digest, "rule config digest"),
        (set_bytes, "point 0 does not recompute"),
        (set_share, "point 0 does not recompute"),
        (set_name, "point 1 does not recompute"),
        (set_top, "summary does not reconcile"),
        (set_sides, "fastest_calls must name every side"),
        (set_measured, "only when all three sources"),
        (set_source_measured, "source one_call_timing label"),
        (set_boundary, "boundary were weakened"),
        (set_definition, "definitions were changed"),
        (set_reason, "reasons were changed"),
        (set_environment, "short tokens"),
        (set_switch, "MPS switches"),
        (set_configs, "other configs"),
        (drop_point, "a gap holds 1 to 16 points"),
    ],
)
def test_every_resealed_tamper_is_caught_by_recomputation(report, mutate, message):
    with pytest.raises(MatmulTiledGapVerificationError, match=message):
        verify_gap_report(resealed(report, mutate))


def test_an_unsealed_change_and_another_rule_config_are_rejected(report):
    changed = copy.deepcopy(report)
    changed["points"][0]["calls_per_sample"] = 2

    with pytest.raises(MatmulTiledGapVerificationError, match="digest"):
        verify_gap_report(changed)
    mapping = RULE.to_mapping()
    mapping["rule"]["near_roof_share_min"] = 0.8
    with pytest.raises(MatmulTiledGapVerificationError, match="supplied rule config"):
        verify_gap_report(report, config=rule_from_mapping(mapping))


def test_only_a_rebuild_from_the_sources_catches_resealed_consistent_inputs(report):
    rebuilt = gap(timing_report(scale={NAIVE_KERNEL: 3}))
    other = gap(timing_report(scale={NAIVE_KERNEL: 2}))

    assert verify_gap_report(report, rebuilt=rebuilt)["sources_checked"] is True
    # A gap from other timings recomputes alone, but not against these sources.
    assert verify_gap_report(other)["sources_checked"] is False
    with pytest.raises(MatmulTiledGapVerificationError, match="supplied source"):
        verify_gap_report(other, rebuilt=rebuilt)


def test_the_gap_verifier_runs_without_torch(tmp_path, report):
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_tiled_gap_verify import verify_gap_report\n"
        f"report = json.loads(open({str(path)!r}).read())\n"
        "print(verify_gap_report(report)['report_sha256'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == report[DIGEST_FIELD]
