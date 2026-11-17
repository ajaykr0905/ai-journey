import itertools
import time

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_report import ENVIRONMENT_FIELDS
from ai_journey.matmul_metal_config import config_from_mapping
from ai_journey.matmul_metal_report import (
    MATMUL_PRECISION_FIELD,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    build_report,
    measure_report,
)
from ai_journey.matmul_metal_source import SHADER_SOURCE, source_sha256
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_config import make_config_mapping
from test_matmul_metal_recorded import RecordedBackend
from test_matmul_metal_runtime import matmul_runtime

SIDES = ("matmul_x_columns", "matmul_x_rows", "torch_matmul")


def scripted_clock():
    return itertools.count(0, 1_000).__next__


def synthetic_report(**changes):
    config = config_from_mapping({**make_config_mapping(), **changes})
    return config, build_report(
        config, backend=RecordedBackend(), clock=scripted_clock()
    )


def test_a_stand_in_run_is_labeled_synthetic_from_top_to_bottom():
    config, report = synthetic_report()

    assert report["kind"] == REPORT_KIND
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["summary"]["label"] == report["label"]
    assert report["config"] == config.to_mapping()
    assert report["config_sha256"] == config.fingerprint()
    assert report["planned_work"] == config.planned_work().to_mapping()
    assert set(report["environment"]) == {*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD}
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["environment"][MATMUL_PRECISION_FIELD] == "highest"


def test_the_report_names_the_float32_matmul_precision_the_backend_ran_with():
    class MediumPrecision(RecordedBackend):
        float32_matmul_precision = "medium"

    config = config_from_mapping(make_config_mapping())

    report = build_report(config, backend=MediumPrecision(), clock=scripted_clock())

    assert report["environment"][MATMUL_PRECISION_FIELD] == "medium"


def test_the_report_records_source_limits_cases_points_and_the_ceiling():
    config, report = synthetic_report()

    shaders = report["shaders"]
    assert (shaders["source"], shaders["source_sha256"]) == (
        SHADER_SOURCE,
        source_sha256(),
    )
    assert [kernel["name"] for kernel in shaders["kernels"]] == list(SIDES[:2])
    assert shaders["kernels"][0]["limits"]["thread_execution_width"] == 32
    correctness = report["correctness"]
    assert len(correctness["cases"]) == 3
    assert correctness["bound"] == "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)"
    assert correctness["sentinel_float32_bits"] == 0x7FC00000
    benchmark = report["benchmark"]
    assert [point["index"] for point in benchmark["points"]] == [0, 1]
    assert benchmark["torch_equivalent"].startswith("torch.matmul(a, b, out=c)")
    # The scripted clock gives every sample 1000 ns, so the point whose batch
    # holds the most FLOPs, 104 calls of 2 * 33 * 17 * 9, sets the ceiling.
    ceiling = benchmark["ceiling"]
    assert ceiling["ceiling_index"] == 0
    assert ceiling["measured_ceiling_gflops_per_second"] == 104 * 10098 / 1000
    run = report["run"]
    samples = [
        sample
        for point in benchmark["points"]
        for side in SIDES
        for sample in point["sides"][side]["samples_ns"]
    ]
    assert run["timed_ns"] == sum(samples) == 1000 * len(samples)
    assert run["timed_ns"] <= run["elapsed_ns"] <= run["max_runtime_ns"]
    assert run["max_runtime_ns"] == 60 * 10**9


def test_the_summary_reconciles_with_the_cases_and_points():
    _, report = synthetic_report()
    summary = report["summary"]
    cases = report["correctness"]["cases"]

    assert summary["correctness_cases"] == 3
    assert summary["naive_outputs_replay_exactly"] is True
    assert summary["max_relative_error"]["matmul_x_rows/mps"] == max(
        case["outputs"][1]["max_relative_error"] for case in cases
    )
    assert summary["max_bound_ratio"] == max(
        output["bound_ratio"] for case in cases for output in case["outputs"]
    )
    assert set(summary["torch_cases_with_other_bits"]) == {
        "torch_matmul/mps",
        "torch_matmul/cpu",
    }
    assert summary["timed_samples"] == 3 * 2 * 3
    assert summary["largest_point"]["m"] == 64
    naive = report["benchmark"]["ceiling"]["naive"]["matmul_x_columns"]
    fraction = summary["naive_fraction_of_measured_ceiling"]["matmul_x_columns"]
    assert fraction == naive["fraction_of_measured_ceiling"]


def test_the_host_timer_on_a_stand_in_backend_is_still_synthetic():
    config = config_from_mapping(make_config_mapping())

    report = build_report(config, backend=RecordedBackend(), clock=time.perf_counter_ns)

    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "time.perf_counter_ns",
    )
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY


def test_a_run_over_its_budget_stops_and_the_seal_covers_every_field():
    budget = {**make_config_mapping()["benchmark"], "max_runtime_seconds": 1}
    config = config_from_mapping({**make_config_mapping(), "benchmark": budget})
    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        build_report(
            config,
            backend=RecordedBackend(),
            clock=itertools.count(0, 10**8).__next__,
        )

    sealed = measure_report(
        config_from_mapping(make_config_mapping()),
        backend=RecordedBackend(),
        clock=scripted_clock(),
    )
    assert sealed[DIGEST_FIELD] == report_digest(sealed)
    with pytest.raises(TypeError, match="MatmulMetalConfig"):
        build_report(make_config_mapping(), backend=RecordedBackend())


@requires_metal
def test_only_the_real_runtime_with_the_host_timer_is_a_measurement():
    config = config_from_mapping(make_config_mapping())

    measured = build_report(config, backend=matmul_runtime())
    injected = build_report(config, backend=matmul_runtime(), clock=scripted_clock())

    assert (measured["measured"], measured["backend"]) == (
        True,
        "torch.mps.compile_shader",
    )
    assert measured["measurement_scope"] == "apple_gpu_metal_this_host"
    assert measured["label"] == "host-specific Apple GPU (Metal) measurement"
    assert measured["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert (injected["measured"], injected["timer"]) == (False, "injected")
    assert injected["backend"] == "torch.mps.compile_shader"
    assert injected["evidence_boundary"] == SYNTHETIC_BOUNDARY
