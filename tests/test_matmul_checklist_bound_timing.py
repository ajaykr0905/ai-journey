import copy
import itertools
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from ai_journey.gpu_timing_clocks import (
    DEVICE_DOMAIN,
    DISPATCH,
    DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION,
    EVENT,
    SYNCHRONIZED,
)
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_checklist_bound_timing import (
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    ChecklistTimingRuntime,
    measure_timing_report,
    method_mapping,
)
from ai_journey.matmul_checklist_bound_timing_verify import verify_timing_report
from ai_journey.matmul_checklist_bound_traffic import (
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
)
from ai_journey.matmul_checklist_config import config_from_mapping
from ai_journey.matmul_checklist_runtime import ChecklistWorkload
from ai_journey.matmul_checklist_variants import VARIANTS, checklist_library
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_gap_timing import FLOOR_INDEX
from ai_journey.matmul_tiled_runtime import SideCheck
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import text_sha256
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest, seal_report
from test_kernel_metal_runtime import darwin, importer, requires_metal
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_correctness import StandInBackend
from test_matmul_checklist_runtime import fake_pipelines

ROOT = Path(__file__).resolve().parents[1]
LOAD = (1.5, 2.25, 3.0)
SIDES = [*VARIANTS, TORCH_MATMUL]
CLOCKS = (EVENT, SYNCHRONIZED, DISPATCH)
EVENT_MS = 0.0005


class EventStandIn:
    """Two stand-in events whose interval is always ``EVENT_MS``."""

    def start(self):
        pass

    def stop(self):
        pass

    def elapsed_ms(self):
        return EVENT_MS


class TimingStandIn(StandInBackend):
    """The checklist NumPy stand-in plus event pairs; each side counts its calls."""

    def __init__(self, *args, disagreement=None, warn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = {}
        self.disagreement = disagreement
        self.warn = warn

    def event_pair(self):
        return EventStandIn()

    def workload(self, left, right, launches):
        bound = super().workload(left, right, launches)
        shape = (left.shape[0], right.shape[1], left.shape[1])

        def counted(name):
            def call():
                self.calls[shape, name] = self.calls.get((shape, name), 0) + 1

            return call

        def check():
            if self.warn:
                warnings.warn("stand-in check warning", UserWarning)
            results = dict(bound.check())
            if self.disagreement is not None:
                results[TORCH_MATMUL] = SideCheck(False, self.disagreement)
            return results

        return ChecklistWorkload(
            sides={name: counted(name) for name in bound.sides},
            synchronize=bound.synchronize,
            check=check,
        )


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def config(**changes):
    return config_from_mapping({**make_config_mapping(), **changes})


def stand_in_timing(backend=None, clock=None, **changes):
    return measure_timing_report(
        config(**changes),
        backend=backend or TimingStandIn(),
        clock=clock or scripted_clock(),
        load_average=lambda: LOAD,
    )


def resealed(report, mutate):
    changed = copy.deepcopy(report)
    del changed[DIGEST_FIELD]
    mutate(changed)
    return seal_report(changed)


@pytest.fixture(scope="module")
def report():
    return stand_in_timing()


def test_a_stand_in_timing_is_synthetic_sealed_and_complete(report):
    summary = verify_timing_report(json.loads(json.dumps(report)), config=config())

    assert report[DIGEST_FIELD] == report_digest(report)
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert summary["label"] == "synthetic timing, not a measurement"
    assert summary["sides"] == SIDES and summary["refused_variants"] == []
    floor, points = report["floor"], report["points"]
    assert [floor["index"], floor["m"], floor["n"], floor["k"]] == [
        FLOOR_INDEX,
        1,
        1,
        1,
    ]
    assert [[point["m"], point["n"], point["k"]] for point in points] == [
        [64, 64, 64],
        [65, 68, 40],
    ]
    for point in (floor, *points):
        assert point["side_order"] == SIDES
        assert list(point["checks"]) == SIDES[1:]
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            assert list(clocks) == list(CLOCKS)
            assert clocks[EVENT]["samples_ns"] == [500, 500, 500]
            assert clocks[SYNCHRONIZED]["timing"]["median_ns"] == 1_000
            assert clocks[DISPATCH]["timing"]["median_ns"] == 1_000
    assert summary["timed_samples"] == 3 * len(SIDES) * 3 * 3
    assert report["summary"]["fastest_host"]["matmul_s7_interior"] == {
        "m": 1,
        "n": 1,
        "k": 1,
        "median_host_ns": 1_000,
    }
    assert report["run"]["host_timed_ns"] == 3 * len(SIDES) * 2 * 3 * 1_000
    assert report["run"]["host_timed_ns"] <= report["run"]["elapsed_ns"]
    assert report["warnings"]["count"] == 0 == summary["warnings"]
    assert report["host_load"]["before"] == list(LOAD)


def test_the_report_seals_the_version_2_event_text_and_both_digests(report):
    texts = report["method"]["domains"]
    kernels = report["kernels"]

    assert report["method"] == method_mapping()
    assert texts == DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[2]
    assert "host time between the two record() calls" in texts[DEVICE_DOMAIN]
    assert "checklist run's point i" in report["method"]["operands"]
    assert kernels["naive_source_sha256"] == source_sha256()
    assert kernels["library_sha256"] == text_sha256(checklist_library())
    assert list(kernels["limits"]) == list(VARIANTS)
    assert [entry["admitted"] for entry in kernels["admissions"]] == [True] * 8
    assert MEASUREMENT_BOUNDARY["hardware_counters_used"] is False
    assert MEASUREMENT_BOUNDARY["occupancy_measured"] is False


def test_each_side_runs_one_call_per_sample_on_every_clock():
    backend = TimingStandIn()

    stand_in_timing(backend)

    # Warmup 1 and repeats 3 on each of 3 clocks: 12 single-call samples.
    for shape in ((1, 1, 1), (64, 64, 64), (65, 68, 40)):
        assert [backend.calls[shape, side] for side in SIDES] == [12] * len(SIDES)


def test_a_refused_kernel_is_recorded_and_never_timed():
    backend = TimingStandIn(threads={"matmul_s6_group_32x32": 512})

    timed = stand_in_timing(backend)
    summary = verify_timing_report(timed)

    refused = [
        entry for entry in timed["kernels"]["admissions"] if not entry["admitted"]
    ]
    assert [(entry["name"], entry["allowed_threads"]) for entry in refused] == [
        ("matmul_s6_group_32x32", 512)
    ]
    assert summary["refused_variants"] == ["matmul_s6_group_32x32"]
    assert "matmul_s6_group_32x32" not in summary["sides"]
    assert all(name != "matmul_s6_group_32x32" for _, name in backend.calls)


def test_warnings_of_the_run_are_sealed_and_counted():
    timed = stand_in_timing(TimingStandIn(warn=True))

    captured = timed["warnings"]
    assert captured["count"] == 3 and captured["distinct"] == 1
    assert captured["records"][0]["message"] == "stand-in check warning"
    assert verify_timing_report(timed)["warnings"] == 3


def test_an_output_past_the_agreement_bound_stops_the_run():
    with pytest.raises(MatmulTiledError, match="torch_matmul sits 1.0"):
        stand_in_timing(TimingStandIn(disagreement=1.0))


def test_a_run_past_its_budget_stops_without_a_report():
    changes = {
        "benchmark": {**make_config_mapping()["benchmark"], "max_runtime_seconds": 1}
    }

    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        stand_in_timing(clock=scripted_clock(10_000_000), **changes)


def test_a_runtime_that_compiled_another_library_cannot_make_the_report():
    runtime = ChecklistTimingRuntime(
        system=darwin, import_module=importer(fake_pipelines()), environ={}
    )
    runtime.library_source = checklist_library().replace("16u", "8u", 1)

    with pytest.raises(MatmulChecklistBoundError, match="another checklist library"):
        measure_timing_report(config(), backend=runtime, load_average=lambda: LOAD)


def set_event(report):
    clocks = report["points"][0]["sides"]["matmul_x_columns"]["clocks"]
    clocks[EVENT]["elapsed_ms"][0] = 0.001


def set_host_sample(report):
    clocks = report["points"][1]["sides"]["matmul_s4_vec4"]["clocks"]
    clocks[SYNCHRONIZED]["samples_ns"][1] = 999


def set_dispatch_above_host(report):
    for point in (report["floor"], *report["points"]):
        clock = point["sides"]["matmul_s5_pad4"]["clocks"][DISPATCH]
        clock["samples_ns"] = [2_000, 2_000, 2_000]
        clock["timing"] = {
            "count": 3,
            "min_ns": 2_000,
            "p25_ns": 2_000,
            "median_ns": 2_000,
            "p75_ns": 2_000,
            "max_ns": 2_000,
            "iqr_ns": 0,
        }


def set_check(report):
    report["points"][0]["checks"][TORCH_MATMUL]["disagreement"] = 1.0


def set_same_bits(report):
    # Equal bits leave no difference, but the stand-in torch side sits 2**-30 away.
    report["floor"]["checks"][TORCH_MATMUL]["same_bits_as_naive"] = True


def set_limits(report):
    limits = report["kernels"]["limits"]["matmul_s3_block_4x4"]
    limits["static_threadgroup_memory_bytes"] = 4096


def set_library(report):
    report["kernels"]["library_sha256"] = source_sha256()


def set_naive_limit(report):
    # The naive group of the config holds 32 * 8 = 256 threads.
    report["kernels"]["limits"]["matmul_x_rows"]["max_threads_per_threadgroup"] = 64


def set_measured(report):
    report["measured"] = True


def set_boundary(report):
    report["evidence_boundary"]["occupancy_measured"] = True


def set_domains(report):
    report["method"]["domains"] = dict(DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[1])


def set_elapsed(report):
    report["run"]["elapsed_ns"] = 10


def set_fastest(report):
    report["summary"]["fastest_host"][TORCH_MATMUL]["median_host_ns"] = 1


def set_refused(report):
    report["summary"]["refused_variants"] = ["matmul_s7_interior"]


def set_warning_count(report):
    report["warnings"]["count"] = 2


def set_config(report):
    report["config"]["seed"] = 78


def set_environment(report):
    report["environment"]["torch_version"] = "torch 2.15"


def set_switch(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_FAST_MATH"] = "1 2"


def set_load(report):
    report["host_load"]["after"] = [1.0, 2.0]


def drop_side(report):
    del report["points"][1]["sides"]["matmul_s6_group_8x8"]


def drop_point(report):
    report["points"].pop()


def set_warmup(report):
    report["floor"]["sides"]["matmul_x_columns"]["warmup"] = 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_event, "does not recompute"),
        (set_host_sample, "does not recompute"),
        (set_dispatch_above_host, "dispatch median exceeds"),
        (set_check, "within the bound"),
        (set_same_bits, "cannot hold the naive bits"),
        (set_limits, "kernel section does not recompute"),
        (set_library, "kernel section does not recompute"),
        (set_naive_limit, "naive group of 256"),
        (set_measured, "only the host timer"),
        (set_boundary, "boundary were weakened"),
        (set_domains, "method texts were changed"),
        (set_elapsed, "run time does not cover"),
        (set_fastest, "summary does not reconcile"),
        (set_refused, "summary does not reconcile"),
        (set_warning_count, "warning counts"),
        (set_config, "config digest"),
        (set_environment, "environment"),
        (set_switch, "MPS switches"),
        (set_load, "host_load"),
        (drop_side, "must time the naive kernel"),
        (drop_point, "do not cover the sweep"),
        (set_warmup, "does not recompute"),
    ],
)
def test_every_resealed_tamper_is_caught_by_recomputation(report, mutate, message):
    with pytest.raises(MatmulChecklistBoundVerificationError, match=message):
        verify_timing_report(resealed(report, mutate))


def test_an_unsealed_change_or_another_config_is_rejected(report):
    changed = copy.deepcopy(report)
    changed["points"][0]["agreement_bound"] = 0.5

    with pytest.raises(MatmulChecklistBoundVerificationError, match="digest"):
        verify_timing_report(changed)
    with pytest.raises(MatmulChecklistBoundVerificationError, match="supplied config"):
        verify_timing_report(report, config=config(seed=78))
    with pytest.raises(MatmulChecklistBoundVerificationError, match="report kind"):
        verify_timing_report(resealed(report, lambda r: r.update(kind="other")))
    with pytest.raises(TypeError, match="MatmulChecklistConfig"):
        verify_timing_report(report, config=make_config_mapping())


def test_the_verifier_runs_without_torch(tmp_path, report):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_checklist_bound_timing_verify import "
        "verify_timing_report\n"
        f"report = json.loads(open({str(path)!r}).read())\n"
        "print(verify_timing_report(report)['report_sha256'])\n"
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


@requires_metal
def test_a_real_one_call_timing_is_a_labeled_apple_gpu_measurement():
    small = config(
        benchmark={
            "shapes": [[65, 68, 40]],
            "warmup": 1,
            "repeats": 3,
            "target_flops_per_sample": 1 << 20,
            "max_runtime_seconds": 60,
        }
    )

    timed = measure_timing_report(small)
    summary = verify_timing_report(timed, config=small)

    assert (timed["measured"], timed["backend"], timed["timer"]) == (
        True,
        "torch.mps.compile_shader",
        "time.perf_counter_ns",
    )
    assert timed["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
    assert summary["sides"] == SIDES and summary["refused_variants"] == []
    for point in (timed["floor"], *timed["points"]):
        assert list(point["checks"]) == SIDES[1:]
        for side in SIDES:
            event = point["sides"][side]["clocks"][EVENT]["timing"]["median_ns"]
            assert event > 0
