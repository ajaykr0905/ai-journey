import copy
import itertools
import json
import os
import subprocess
import sys
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
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_metal_source import source_sha256
from ai_journey.matmul_tiled_config import config_from_mapping
from ai_journey.matmul_tiled_gap_timing import (
    FLOOR_INDEX,
    MEASUREMENT_BOUNDARY,
    REPORT_SCHEMA_VERSION,
    SCHEMA_VERSIONS,
    SYNTHETIC_BOUNDARY,
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
    TiledTimingRuntime,
    measure_timing_report,
    method_mapping,
)
from ai_journey.matmul_tiled_gap_timing_verify import verify_timing_report
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL, SideCheck, TiledWorkload
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import VARIANTS, library_source, text_sha256
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest, seal_report
from test_kernel_metal_runtime import darwin, importer, requires_metal
from test_matmul_tiled_config import make_config_mapping
from test_matmul_tiled_correctness import StandInBackend
from test_matmul_tiled_runtime import fake_pipelines

ROOT = Path(__file__).resolve().parents[1]
LOAD = (1.5, 2.25, 3.0)
DAY_76 = tuple(VARIANTS.values())
SIDES = [NAIVE_KERNEL, *VARIANTS, TORCH_MATMUL]
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
    """The tiled NumPy stand-in plus event pairs; every side counts its calls."""

    def __init__(self, *args, disagreement=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = {}
        self.disagreement = disagreement

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
            results = dict(bound.check())
            if self.disagreement is not None:
                results[TORCH_MATMUL] = SideCheck(False, self.disagreement)
            return results

        return TiledWorkload(
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
    assert summary["sides"] == SIDES
    floor, points = report["floor"], report["points"]
    assert (floor["index"], floor["m"], floor["n"], floor["k"]) == (
        FLOOR_INDEX,
        1,
        1,
        1,
    )
    assert [[point["m"], point["n"], point["k"]] for point in points] == [
        [64, 64, 64],
        [40, 24, 72],
    ]
    for point in (floor, *points):
        assert point["side_order"] == SIDES
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            assert list(clocks) == list(CLOCKS)
            assert clocks[EVENT]["samples_ns"] == [500, 500, 500]
            assert clocks[EVENT]["elapsed_ms"] == [EVENT_MS] * 3
            assert clocks[SYNCHRONIZED]["timing"]["median_ns"] == 1_000
            assert clocks[DISPATCH]["timing"]["median_ns"] == 1_000
    assert summary["timed_samples"] == 3 * 5 * 3 * 3
    assert report["summary"]["fastest_host"][NAIVE_KERNEL] == {
        "m": 1,
        "n": 1,
        "k": 1,
        "median_host_ns": 1_000,
    }
    assert report["host_load"]["before"] == list(LOAD)
    assert report["run"]["host_timed_ns"] == 3 * 5 * 2 * 3 * 1_000
    assert report["run"]["host_timed_ns"] <= report["run"]["elapsed_ns"]


def test_the_kernels_section_names_both_libraries_and_every_verdict(report):
    kernels = report["kernels"]

    assert kernels["naive_source_sha256"] == source_sha256()
    assert kernels["tiled_source_sha256"] == text_sha256(library_source(DAY_76))
    assert list(kernels["limits"]) == [NAIVE_KERNEL, *VARIANTS]
    assert [entry["admitted"] for entry in kernels["admissions"]] == [True] * 3


def test_each_side_runs_one_call_per_sample_on_every_clock():
    backend = TimingStandIn()

    stand_in_timing(backend)

    # Warmup 1 and repeats 3 on each of 3 clocks: 12 single-call samples.
    for shape in ((1, 1, 1), (64, 64, 64), (40, 24, 72)):
        assert [backend.calls[shape, side] for side in SIDES] == [12] * 5


def test_a_refused_tile_is_recorded_and_never_timed():
    backend = TimingStandIn(threads={"matmul_tiled_32": 512})

    timed = stand_in_timing(backend)
    summary = verify_timing_report(timed)

    refused = timed["kernels"]["admissions"][2]
    assert (refused["name"], refused["admitted"], refused["allowed_threads"]) == (
        "matmul_tiled_32",
        False,
        512,
    )
    assert summary["sides"] == [NAIVE_KERNEL, "matmul_tiled_8", "matmul_tiled_16"] + [
        TORCH_MATMUL
    ]
    assert all(name != "matmul_tiled_32" for _, name in backend.calls)


def test_an_output_past_the_agreement_bound_stops_the_run():
    with pytest.raises(MatmulTiledError, match="torch_matmul sits 1.0"):
        stand_in_timing(TimingStandIn(disagreement=1.0))


def test_a_run_past_its_budget_stops_without_a_report():
    changes = {
        "benchmark": {**make_config_mapping()["benchmark"], "max_runtime_seconds": 1}
    }

    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        stand_in_timing(clock=scripted_clock(10_000_000), **changes)


def test_a_runtime_that_compiled_other_tiles_cannot_make_the_report():
    runtime = TiledTimingRuntime(
        DAY_76[:2],
        system=darwin,
        import_module=importer(fake_pipelines()),
        environ={},
    )

    with pytest.raises(MatmulTiledGapError, match="another tiled library"):
        measure_timing_report(config(), backend=runtime, load_average=lambda: LOAD)


def set_event(report):
    report["points"][0]["sides"][NAIVE_KERNEL]["clocks"][EVENT]["elapsed_ms"][0] = 0.001


def set_host_sample(report):
    clocks = report["points"][1]["sides"]["matmul_tiled_8"]["clocks"]
    clocks[SYNCHRONIZED]["samples_ns"][1] = 999


def set_median(report):
    clocks = report["floor"]["sides"][TORCH_MATMUL]["clocks"]
    clocks[DISPATCH]["timing"]["median_ns"] = 900


def set_dispatch_above_host(report):
    for point in (report["floor"], *report["points"]):
        clock = point["sides"][NAIVE_KERNEL]["clocks"][DISPATCH]
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


def set_limits(report):
    report["kernels"]["limits"]["matmul_tiled_16"]["max_threads_per_threadgroup"] = 128


def set_measured(report):
    report["measured"] = True


def set_boundary(report):
    report["evidence_boundary"]["cublas_measured"] = True


def set_method(report):
    report["method"]["floor"] = "no floor"


def set_elapsed(report):
    report["run"]["elapsed_ns"] = 10


def set_fastest(report):
    report["summary"]["fastest_host"][TORCH_MATMUL]["median_host_ns"] = 1


def set_config(report):
    report["config"]["seed"] = 77


def set_environment(report):
    report["environment"]["torch_version"] = "torch 2.15"


def set_switch(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_FAST_MATH"] = "1 2"


def set_load(report):
    report["host_load"]["after"] = [1.0, 2.0]


def drop_side(report):
    del report["points"][1]["sides"]["matmul_tiled_16"]


def set_warmup(report):
    report["floor"]["sides"][NAIVE_KERNEL]["warmup"] = 0


def set_same_bits(report):
    # Equal bits leave no difference, but the stand-in torch side sits 2**-30 away.
    report["floor"]["checks"][TORCH_MATMUL]["same_bits_as_naive"] = True


def set_naive_limit(report):
    # The naive group of the config holds 32 * 8 = 256 threads.
    report["kernels"]["limits"][NAIVE_KERNEL]["max_threads_per_threadgroup"] = 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_event, "does not recompute"),
        (set_host_sample, "does not recompute"),
        (set_median, "does not recompute"),
        (set_dispatch_above_host, "dispatch median exceeds"),
        (set_check, "within the bound"),
        (set_limits, "kernel section does not recompute"),
        (set_measured, "only the host timer"),
        (set_boundary, "boundary were weakened"),
        (set_method, "method texts"),
        (set_elapsed, "run time does not cover"),
        (set_fastest, "summary does not reconcile"),
        (set_config, "config digest"),
        (set_environment, "environment"),
        (set_switch, "MPS switches"),
        (set_load, "host_load"),
        (drop_side, "must time the naive kernel"),
        (set_warmup, "does not recompute"),
        (set_same_bits, "cannot hold the naive bits"),
        (set_naive_limit, "naive group"),
    ],
)
def test_every_resealed_tamper_is_caught_by_recomputation(report, mutate, message):
    with pytest.raises(MatmulTiledGapVerificationError, match=message):
        verify_timing_report(resealed(report, mutate))


def test_an_unsealed_change_or_another_config_is_rejected(report):
    changed = copy.deepcopy(report)
    changed["points"][0]["agreement_bound"] = 0.5

    with pytest.raises(MatmulTiledGapVerificationError, match="digest"):
        verify_timing_report(changed)
    other = config(seed=77)
    with pytest.raises(MatmulTiledGapVerificationError, match="supplied config"):
        verify_timing_report(report, config=other)


def test_the_verifier_runs_without_torch(tmp_path, report):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_tiled_gap_timing_verify import verify_timing_report\n"
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


def as_schema(report, version, domains):
    """The same run resealed as a report of ``version`` with other domain texts."""

    def mutate(changed):
        changed["schema_version"] = version
        changed["method"]["domains"] = dict(
            DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[domains]
        )

    return resealed(report, mutate)


def test_a_new_report_seals_schema_2_with_the_host_time_event_text(report):
    texts = report["method"]["domains"]

    summary = verify_timing_report(report, config=config())

    assert summary["report_sha256"] == report[DIGEST_FIELD]
    assert report["schema_version"] == REPORT_SCHEMA_VERSION == 2
    assert SCHEMA_VERSIONS == (1, 2)
    assert report["method"] == method_mapping(2)
    assert texts == DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[2]
    assert "device-side" not in texts[DEVICE_DOMAIN]
    assert "host time between the two record() calls" in texts[DEVICE_DOMAIN]


def test_a_schema_1_report_verifies_with_its_device_side_text(report):
    old = as_schema(report, 1, 1)
    method, new = old["method"], report["method"]

    summary = verify_timing_report(old, config=config())

    assert summary["report_sha256"] == old[DIGEST_FIELD] != report[DIGEST_FIELD]
    assert method == method_mapping(1)
    assert method["domains"][DEVICE_DOMAIN].startswith("device-side: ")
    # Only the event-domain text differs between the two versions.
    assert {name for name in method if method[name] != new[name]} == {"domains"}
    domains = method["domains"]
    assert {name for name in domains if domains[name] != new["domains"][name]} == {
        DEVICE_DOMAIN
    }


@pytest.mark.parametrize(
    ("version", "other"), [(1, 2), (2, 1)], ids=["schema-1", "schema-2"]
)
def test_each_schema_version_refuses_the_domain_texts_of_the_other(
    report, version, other
):
    with pytest.raises(
        MatmulTiledGapVerificationError,
        match=f"method texts are not those of schema_version {version}$",
    ):
        verify_timing_report(as_schema(report, version, other))


@pytest.mark.parametrize("version", [0, 3, True, 2.0])
def test_an_unknown_schema_version_is_refused(report, version):
    def mutate(changed):
        changed["schema_version"] = version

    with pytest.raises(MatmulTiledGapVerificationError, match="must be one of 1, 2$"):
        verify_timing_report(resealed(report, mutate))
    with pytest.raises(ValueError, match="no one-call timing schema_version"):
        method_mapping(version)


@requires_metal
def test_a_real_one_call_timing_is_a_labeled_apple_gpu_measurement():
    small = config(
        benchmark={
            "shapes": [[64, 64, 64]],
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
    assert summary["sides"] == SIDES
    for point in (timed["floor"], *timed["points"]):
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            assert all(value > 0 for value in clocks[EVENT]["elapsed_ms"])
            assert (
                clocks[DISPATCH]["timing"]["median_ns"]
                <= clocks[SYNCHRONIZED]["timing"]["median_ns"]
            )
        assert all(
            check["disagreement"] <= point["agreement_bound"]
            for check in point["checks"].values()
        )
