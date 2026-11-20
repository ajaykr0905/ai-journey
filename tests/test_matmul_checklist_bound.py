import copy
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from ai_journey.gpu_ceiling_roofline import MeasuredRoofline
from ai_journey.kernel_metal_report import SYNTHETIC_LABEL
from ai_journey.matmul_bottleneck_config import config_from_mapping as rule_from_mapping
from ai_journey.matmul_bottleneck_config import load_config as load_rule_config
from ai_journey.matmul_bottleneck_rule import PointInputs, classify
from ai_journey.matmul_checklist_bound import (
    CEILING_SOURCE,
    CHECKLIST_SOURCE,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    TIMING_SOURCE,
    build_bound_report,
)
from ai_journey.matmul_checklist_bound_io import (
    BoundSources,
    load_bound_report,
    save_bound_report,
)
from ai_journey.matmul_checklist_bound_timing import measure_timing_report
from ai_journey.matmul_checklist_bound_traffic import (
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
    device_bytes,
    intensity,
    kernel_traffic,
)
from ai_journey.matmul_checklist_bound_verify import verify_bound_report
from ai_journey.matmul_checklist_config import config_from_mapping
from ai_journey.matmul_checklist_runtime import ChecklistWorkload
from ai_journey.matmul_checklist_steps import CHECKLIST, checklist_sha256
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from ai_journey.memory_access_tiling import MatmulShape
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_matmul_checklist_bound_timing import LOAD, TimingStandIn
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_correctness import StandInBackend
from test_matmul_checklist_recorded import RECORDED_CONFIG
from test_matmul_checklist_recorded import RECORDED_REPORT_JSON as CHECKLIST_JSON
from test_matmul_checklist_report import stand_in_report
from test_matmul_tiled_gap import ClockEvents, VirtualClock

ROOT = Path(__file__).resolve().parents[1]
RULE = load_rule_config(ROOT / "config/day-75-matmul-bottleneck.json")
SIDES = [*VARIANTS, TORCH_MATMUL]
FLOOR_NS = 20_000


def checklist():
    return json.loads(CHECKLIST_JSON)


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

        return ChecklistWorkload(
            sides={name: paced(name) for name in bound.sides},
            synchronize=bound.synchronize,
            check=bound.check,
        )


def timing_report(scale=None, environment=None, mapping=None, **backend_options):
    """A synthetic timing of the recorded checklist config, on the recorded host.

    The stand-in reads the environment of the test host, so the report is
    pinned to the recorded host's tokens and sealed again. It stays
    synthetic, as its backend and timer say.
    """

    clock = VirtualClock()
    backend = PacedStandIn(clock, steady_costs(checklist(), scale), **backend_options)
    report = measure_timing_report(
        config_from_mapping(mapping or RECORDED_CONFIG),
        backend=backend,
        clock=clock,
        load_average=lambda: LOAD,
    )
    del report[DIGEST_FIELD]
    report["environment"] = dict(environment or checklist()["environment"])
    return seal_report(report)


def bound(timing=None, rule=RULE):
    return build_bound_report(checklist(), ceilings(), timing or timing_report(), rule)


def resealed(report, mutate):
    changed = copy.deepcopy(report)
    del changed[DIGEST_FIELD]
    mutate(changed)
    return seal_report(changed)


@pytest.fixture(scope="module")
def report():
    return bound(timing_report(scale={"matmul_x_rows": 3}))


def test_a_bound_joins_three_verified_sources_and_verifies_alone(report):
    summary = verify_bound_report(json.loads(json.dumps(report)), config=RULE)

    assert (report["measured"], report["evidence_boundary"]) == (
        False,
        SYNTHETIC_BOUNDARY,
    )
    sources = report["sources"]
    assert sources[CHECKLIST_SOURCE]["report_sha256"] == checklist()[DIGEST_FIELD]
    assert sources[CHECKLIST_SOURCE]["measured"] is True
    assert sources[CEILING_SOURCE]["report_sha256"] == ceilings()[DIGEST_FIELD]
    assert sources[TIMING_SOURCE]["measured"] is False
    assert (
        sources[CEILING_SOURCE]["load_average_before"]
        == (ceilings()["host_load"]["before"])
    )
    assert report["sides"] == SIDES == summary["sides"]
    assert [[point["m"], point["n"], point["k"]] for point in report["points"]] == [
        [256, 256, 256],
        [200, 300, 100],
    ]
    assert report["checklist_sha256"] == checklist_sha256()
    assert report["comparison"]["min_faster_ratio"] == 1.1
    assert report["rule"]["config_sha256"] == RULE.fingerprint()
    assert report["fastest_calls"]["matmul_s7_interior"]["m"] == 1
    assert summary["sources_checked"] is False
    assert MEASUREMENT_BOUNDARY["inference_from_timings"] is True
    assert MEASUREMENT_BOUNDARY["occupancy_measured"] is False


def test_each_kernel_side_carries_its_traffic_model_and_attainable_rate(report):
    fma = report["ceilings"]["fma_gflops_per_second"]
    bandwidth = report["ceilings"]["bandwidth_gb_per_second"]
    roofline = MeasuredRoofline(fma, bandwidth, "0" * 64, False, SYNTHETIC_LABEL)

    for point in report["points"]:
        shape = MatmulShape(point["m"], point["n"], point["k"])
        for side in SIDES:
            entry = point["sides"][side]
            traffic = entry["traffic"]
            if side == TORCH_MATMUL:
                assert traffic is None
                continue
            model = kernel_traffic(side, shape)
            rate = entry["median_gflops_per_second"]
            value = float(intensity(model))
            expected = roofline.below_ceiling(rate, value)
            roof = traffic["roof"]
            assert {key: traffic[key] for key in traffic if key != "roof"} == (
                model.to_mapping()
            )
            assert (
                roof["attainable_gflops_per_second"]
                == (expected["attainable_gflops_per_second"])
            )
            assert roof["attainable_share"] == expected["fraction_of_ceiling"]
            assert roof["bound"] == expected["bound"]
            assert roof["modeled_device_gb_per_second"] == pytest.approx(
                device_bytes(model) / (shape.flops / rate)
            )
            assert entry["fma_ceiling_share"] == rate / fma
    point = report["points"][0]["sides"]
    assert point["matmul_x_columns"]["traffic"]["roof"]["attainable_share"] > 1
    assert point["matmul_x_columns"]["traffic"]["threadgroup"] is None
    reads = point["matmul_s7_interior"]["traffic"]["threadgroup"]["read_elements"]
    rate = point["matmul_s7_interior"]["median_gflops_per_second"]
    assert point["matmul_s7_interior"]["traffic"]["roof"][
        "modeled_threadgroup_reads_per_ns"
    ] == pytest.approx(reads * rate / (2 * 256**3))
    assert report["summary"]["points_above_modeled_roof"]["matmul_x_columns"] == 2
    assert report["points"][0]["ideal_reuse"]["bytes"] == 4 * 3 * 256 * 256


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
    # One call of matmul_x_rows took three steady times, the others one.
    tops = {side: report["summary"]["top_bottleneck"][side]["top"] for side in SIDES}
    assert tops == {
        **{side: "latency or occupancy-bound" for side in SIDES},
        "matmul_x_rows": "dispatch-bound",
    }


def test_each_step_row_sets_the_modeled_change_beside_the_checklist_ratio(report):
    rows = {row["variant"]: row for row in report["steps"]}
    speedup = checklist()["speedup"]["points"]

    assert list(rows) == [name for step in CHECKLIST for name in step.variants]
    for step in CHECKLIST:
        for variant in step.variants:
            row = rows[variant]
            assert (row["step"], row["base"]) == (step.step, step.base)
            assert row["predicted"] == list(step.predicted(variant))
            for entry, source in zip(row["points"], speedup):
                recorded = next(
                    item for item in source["rows"] if item["variant"] == variant
                )
                assert entry["measured"]["ratio"] == recorded["to_base"]["ratio"]
                assert entry["measured"]["verdict"] == recorded["to_base"]["verdict"]
                assert entry["measured"]["held"] is recorded["held"]
                shape = MatmulShape(entry["m"], entry["n"], entry["k"])
                first = kernel_traffic(variant, shape)
                second = kernel_traffic(step.base, shape)
                ratio = entry["model"]["device_bytes_ratio"]
                assert Fraction(ratio["numerator"], ratio["denominator"]) == Fraction(
                    device_bytes(first), device_bytes(second)
                )
    s2 = rows["matmul_s2_coarse_4x1"]["points"][0]["model"]
    assert s2["device_bytes_ratio"]["decimal"] == "1.000000"
    assert s2["threadgroup_reads_per_fma"]["base"]["decimal"] == "2.000000"
    assert s2["threadgroup_reads_per_fma"]["variant"]["decimal"] == "1.250000"
    s4 = rows["matmul_s4_vec4"]["points"][0]["model"]
    assert s4["load_operations_ratio"]["decimal"] == "0.250000"
    s1 = rows["matmul_tiled_32"]["points"][0]["model"]
    assert s1["threadgroup_reads_per_fma"] == {
        "base": None,
        "variant": {"numerator": 2, "denominator": 1, "decimal": "2.000000"},
    }
    assert report["summary"]["step_verdicts"]["matmul_x_columns"] == ["faster"] * 2


def test_a_refused_kernel_keeps_its_model_but_has_no_measured_values():
    refused = "matmul_s6_group_32x32"
    source = stand_in_report(StandInBackend(threads={refused: 512}))
    del source[DIGEST_FIELD]
    source["environment"] = dict(checklist()["environment"])
    timing = timing_report(mapping=make_config_mapping(), threads={refused: 512})

    report = build_bound_report(seal_report(source), ceilings(), timing, RULE)
    rows = {row["variant"]: row for row in report["steps"]}

    assert refused not in report["sides"]
    assert verify_bound_report(report)["sides"] == report["sides"]
    for entry in rows[refused]["points"]:
        assert entry["measured"] is None
        assert entry["fma_ceiling_share"]["variant"] is None
        assert entry["bottleneck"]["variant"] is None
        assert entry["bottleneck"]["base"] is not None
        assert entry["model"]["device_bytes_ratio"]["numerator"] > 0
    assert report["summary"]["step_verdicts"][refused] == [None, None]
    assert refused not in report["summary"]["modeled_roof_share_range"]


def test_a_rule_config_with_another_roof_is_refused():
    mapping = RULE.to_mapping()
    mapping["rule"]["intensity_model"] = "no_reuse"

    with pytest.raises(MatmulChecklistBoundError, match="ideal_reuse roof"):
        bound(rule=rule_from_mapping(mapping))


def other_switch(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_FAST_MATH"] = "1"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"environment": {**checklist()["environment"], "torch_version": "2.14"}},
            "different software",
        ),
        (
            {
                "environment": {
                    **checklist()["environment"],
                    "torch_float32_matmul_precision": "high",
                }
            },
            "precisions",
        ),
        ({"mapping": {**RECORDED_CONFIG, "seed": 78}}, "checklist report's config"),
        ({"threads": {"matmul_s6_group_32x32": 512}}, "admitted other kernels"),
    ],
)
def test_sources_that_do_not_pair_are_refused(options, message):
    with pytest.raises(MatmulChecklistBoundError, match=message):
        bound(timing_report(**options))


def test_sources_with_other_mps_switches_are_refused():
    timing = resealed(timing_report(), other_switch)

    with pytest.raises(MatmulChecklistBoundError, match="MPS switches"):
        bound(timing)


def test_a_tampered_source_is_rejected_by_its_own_verifier():
    timing = timing_report()
    tampered_checklist = {**checklist(), "label": SYNTHETIC_LABEL}
    tampered_ceiling = {**ceilings(), "measured": False}
    tampered_timing = {**timing, "measured": True}

    for arguments, name in (
        ((tampered_checklist, ceilings(), timing), "checklist report"),
        ((checklist(), tampered_ceiling, timing), "gpu-ceilings report"),
        ((checklist(), ceilings(), tampered_timing), "one-call timing report"),
    ):
        with pytest.raises(MatmulChecklistBoundVerificationError, match=name):
            build_bound_report(*arguments, RULE)


def set_median(report):
    report["points"][0]["sides"]["matmul_s4_vec4"]["median_sample_ns"] += 1


def set_quartile(report):
    # The IQR ranges of S0 and its base now overlap, so S0 is not clearly faster.
    report["points"][1]["sides"]["matmul_x_columns"]["p75_sample_ns"] = 2_000_000


def set_order(report):
    entry = report["points"][1]["sides"]["matmul_s5_pad4"]
    entry["p25_sample_ns"] = entry["p75_sample_ns"] + 1


def set_host(report):
    report["points"][1]["sides"][TORCH_MATMUL]["one_call"]["median_host_ns"] += 7


def set_fastest(report):
    report["fastest_calls"]["matmul_s3_block_4x4"]["median_host_ns"] = 10**9


def set_fma(report):
    report["ceilings"]["fma_gflops_per_second"] *= 2


def set_ridge(report):
    report["ceilings"]["ridge_flops_per_byte"] = 1.0


def set_threshold(report):
    report["rule"]["config"]["rule"]["near_roof_share_min"] = 0.01


def set_minimum(report):
    # S0 ran 2.2 to 2.6 times as fast as its base, below this minimum.
    report["comparison"]["min_faster_ratio"] = 3.5


def set_ratio_rule(report):
    report["comparison"]["ratio_rule"] = "faster when the median is lower"


def set_bytes(report):
    device = report["points"][0]["sides"]["matmul_s6_group_8x8"]["traffic"]["device"]
    device["bytes"] += 4


def set_reads(report):
    shared = report["points"][0]["sides"]["matmul_s2_coarse_4x1"]["traffic"][
        "threadgroup"
    ]
    shared["reads_per_fma"] = {"numerator": 1, "denominator": 2, "decimal": "0.500000"}


def set_share(report):
    roof = report["points"][0]["sides"]["matmul_x_columns"]["traffic"]["roof"]
    roof["attainable_share"] = 0.5


def set_name(report):
    decision = report["points"][1]["sides"]["matmul_s7_interior"]["decision"]
    decision["bottleneck"] = "memory-bound"


def set_verdict(report):
    entry = report["steps"][4]["points"][0]
    entry["measured"]["verdict"] = "faster"


def set_held(report):
    report["steps"][0]["points"][1]["measured"]["held"] = False


def set_step_model(report):
    model = report["steps"][3]["points"][0]["model"]
    model["device_bytes_ratio"] = {"numerator": 1, "denominator": 2, "decimal": "0.5"}


def set_top(report):
    report["summary"]["top_bottleneck"]["matmul_x_rows"]["top"] = "compute-bound"


def set_step_verdicts(report):
    report["summary"]["step_verdicts"]["matmul_s5_pad4"] = ["slower", "slower"]


def set_sides(report):
    report["sides"] = ["matmul_x_columns", "matmul_s7_interior", TORCH_MATMUL]


def set_order_of_sides(report):
    report["sides"] = [SIDES[0], *reversed(SIDES[1:-1]), TORCH_MATMUL]


def set_measured(report):
    report["measured"] = True


def set_source_measured(report):
    report["sources"][TIMING_SOURCE]["measured"] = True


def set_source_kind(report):
    report["sources"][CHECKLIST_SOURCE]["kind"] = "matmul_tiled_report"


def set_boundary(report):
    report["evidence_boundary"]["occupancy_measured"] = True


def set_definition(report):
    report["definitions"]["decision"] = "a name from hardware counters"


def set_model_text(report):
    report["traffic_model"] = "measured bytes"


def set_reason(report):
    report["reasons"]["below_both_roofs"]["bottleneck"] = "memory-bound"


def set_plan(report):
    report["checklist_sha256"] = "0" * 64


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
        (set_quartile, "steps do not recompute"),
        (set_order, "ordered sample times"),
        (set_host, "point 1 does not recompute"),
        (set_fastest, "faster than the fastest call"),
        (set_fma, "ceilings section does not recompute"),
        (set_ridge, "ceilings section does not recompute"),
        (set_threshold, "rule config digest"),
        (set_minimum, "steps do not recompute"),
        (set_ratio_rule, "ratio rule"),
        (set_bytes, "point 0 does not recompute"),
        (set_reads, "point 0 does not recompute"),
        (set_share, "point 0 does not recompute"),
        (set_name, "point 1 does not recompute"),
        (set_verdict, "steps do not recompute"),
        (set_held, "steps do not recompute"),
        (set_step_model, "steps do not recompute"),
        (set_top, "summary does not reconcile"),
        (set_step_verdicts, "summary does not reconcile"),
        (set_sides, "fastest_calls must name every side"),
        (set_order_of_sides, "registered kernels in order"),
        (set_measured, "only when all three sources"),
        (set_source_measured, "source one_call_timing label"),
        (set_source_kind, "another kind"),
        (set_boundary, "boundary were weakened"),
        (set_definition, "definitions were changed"),
        (set_model_text, "traffic model text"),
        (set_reason, "reasons were changed"),
        (set_plan, "declared checklist"),
        (set_environment, "short tokens"),
        (set_switch, "MPS switches"),
        (set_configs, "other configs"),
        (drop_point, "a bound holds 1 to 16 points"),
    ],
)
def test_every_resealed_tamper_is_caught_by_recomputation(report, mutate, message):
    with pytest.raises(MatmulChecklistBoundVerificationError, match=message):
        verify_bound_report(resealed(report, mutate))


def test_an_unsealed_change_and_another_rule_config_are_rejected(report):
    changed = copy.deepcopy(report)
    changed["points"][0]["calls_per_sample"] = 2

    with pytest.raises(MatmulChecklistBoundVerificationError, match="digest"):
        verify_bound_report(changed)
    mapping = RULE.to_mapping()
    mapping["rule"]["near_roof_share_min"] = 0.8
    with pytest.raises(
        MatmulChecklistBoundVerificationError, match="supplied rule config"
    ):
        verify_bound_report(report, config=rule_from_mapping(mapping))
    with pytest.raises(TypeError, match="BottleneckConfig"):
        verify_bound_report(report, config=mapping)


def test_only_a_rebuild_from_the_sources_catches_resealed_consistent_inputs(report):
    rebuilt = bound(timing_report(scale={"matmul_x_rows": 3}))
    other = bound(timing_report(scale={"matmul_x_rows": 2}))

    assert verify_bound_report(report, rebuilt=rebuilt)["sources_checked"] is True
    # A bound from other timings recomputes alone, but not against these sources.
    assert verify_bound_report(other)["sources_checked"] is False
    with pytest.raises(MatmulChecklistBoundVerificationError, match="supplied source"):
        verify_bound_report(other, rebuilt=rebuilt)


def test_the_bound_verifier_runs_without_torch(tmp_path, report):
    path = tmp_path / "bound.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_checklist_bound_verify import verify_bound_report\n"
        f"report = json.loads(open({str(path)!r}).read())\n"
        "print(verify_bound_report(report)['report_sha256'])\n"
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


def test_the_loader_reads_canonical_bytes_and_needs_the_rule_to_rebuild(
    tmp_path, report
):
    path = tmp_path / "bound.json"
    assert save_bound_report(path, report, config=RULE)["sources_checked"] is False

    loaded, summary = load_bound_report(path, config=RULE)
    assert loaded == report and summary["report_sha256"] == report[DIGEST_FIELD]
    with pytest.raises(MatmulChecklistBoundError, match="needs the rule config"):
        load_bound_report(path, sources=BoundSources(path, path, path))
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with pytest.raises(MatmulChecklistBoundVerificationError, match="not canonical"):
        load_bound_report(pretty)
    with pytest.raises(TypeError, match="pathlib.Path"):
        BoundSources(str(path), path, path)
    with pytest.raises(MatmulChecklistBoundError, match="alias"):
        save_bound_report(path, report, protected=(path,))
