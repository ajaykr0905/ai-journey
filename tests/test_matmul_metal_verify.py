import copy
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy
import pytest

from ai_journey.kernel_metal_report import (
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_SCOPE,
)
from ai_journey.matmul_metal_config import config_from_mapping
from ai_journey.matmul_metal_report import MEASUREMENT_BOUNDARY, measure_report
from ai_journey.matmul_metal_verify import (
    MatmulMetalVerificationError,
    _replay,
    verify_report,
)
from ai_journey.memory_access_benchmark import TIMER
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from test_matmul_metal_config import make_config_mapping
from test_matmul_metal_recorded import RECORDED_REPORT_JSON, RecordedBackend
from test_matmul_metal_reference import TRAPS, exact_fma, trap_operands

ROOT = Path(__file__).resolve().parents[1]


def synthetic_report(**changes):
    config = config_from_mapping({**make_config_mapping(), **changes})
    clock = itertools.count(0, 1_000).__next__
    return config, measure_report(config, backend=RecordedBackend(), clock=clock)


def resealed(report, change):
    tampered = copy.deepcopy(report)
    del tampered[DIGEST_FIELD]
    change(tampered)
    tampered[DIGEST_FIELD] = report_digest(tampered)
    return tampered


def test_a_synthetic_report_verifies_as_synthetic_and_binds_to_its_config():
    config, report = synthetic_report()

    summary = verify_report(report, config=config)

    assert summary["report_sha256"] == report[DIGEST_FIELD]
    assert summary["config_sha256"] == config.fingerprint()
    assert (summary["measured"], summary["scope"]) == (
        False,
        "synthetic_backend_or_clock",
    )
    assert summary["label"] == "synthetic timing, not a measurement"
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (3, 2)
    assert summary["timed_samples"] == 18
    assert 0 < summary["max_bound_ratio"] < 1
    other = config_from_mapping({**make_config_mapping(), "name": "other-config"})
    with pytest.raises(MatmulMetalVerificationError, match="supplied config"):
        verify_report(report, config=other)


def test_tied_points_resolve_to_the_first_one_as_the_run_chose():
    # 64**3 and 32 x 64 x 128 have the same FLOPs, so with every sample at
    # 1000 ns the two points tie on every rate and on the largest point.
    tied = {
        **make_config_mapping()["benchmark"],
        "shapes": [[64, 64, 64], [32, 64, 128]],
    }
    config, report = synthetic_report(benchmark=tied)

    summary = verify_report(report, config=config)

    first, second = report["benchmark"]["points"]
    for side in ("matmul_x_columns", "matmul_x_rows", "torch_matmul"):
        rates = [
            point["sides"][side]["median_gflops_per_second"]
            for point in (first, second)
        ]
        assert rates[0] == rates[1], side
    ceiling = report["benchmark"]["ceiling"]
    assert ceiling["ceiling_index"] == 0
    assert {entry["best_index"] for entry in ceiling["naive"].values()} == {0}
    assert (report["summary"]["largest_point"]["m"], summary["measured"]) == (64, False)


def first_case(report):
    return report["correctness"]["cases"][1]


def first_point(report):
    return report["benchmark"]["points"][0]


def flip_differing(output):
    """Claim other bits where the digest says none, or none where it says some."""

    output["differing_elements"] = int(output["differing_elements"] == 0)


def shorten_first_sample(side):
    side["samples_ns"][0] = 7


def overcount_differing(report):
    """Claim more differing elements than C holds, with a summary to match."""

    case = first_case(report)
    case["outputs"][2].update(
        buffer_sha256="0" * 64, differing_elements=case["m"] * case["n"] + 1
    )
    report["summary"]["torch_cases_with_other_bits"]["torch_matmul/mps"] = sum(
        entry["outputs"][2]["differing_elements"] > 0
        for entry in report["correctness"]["cases"]
    )


def raise_both_thread_limits(report):
    # A launch depends only on the SIMD width, so it still rebuilds.
    for kernel in report["shaders"]["kernels"]:
        kernel["limits"]["max_threads_per_threadgroup"] = 2048


TAMPERS = {
    "naive digest": (
        lambda r: first_case(r)["outputs"][1].update(buffer_sha256="0" * 64),
        "matmul_x_rows does not rebuild from the replay",
    ),
    "naive error": (
        lambda r: first_case(r)["outputs"][0].update(max_relative_error=0.0),
        "matmul_x_columns does not rebuild",
    ),
    "naive differing": (
        lambda r: first_case(r)["outputs"][0].update(differing_elements=1),
        "matmul_x_columns does not rebuild",
    ),
    "torch error above the bound": (
        lambda r: first_case(r)["outputs"][3].update(max_relative_error=1.0),
        "error within the bound",
    ),
    "torch count against its digest": (
        lambda r: flip_differing(first_case(r)["outputs"][2]),
        "differing elements consistently",
    ),
    "torch ratio": (
        lambda r: first_case(r)["outputs"][2].update(bound_ratio=0.5),
        "torch_matmul on mps is malformed",
    ),
    "case reference": (
        lambda r: first_case(r).update(reference_sha256="f" * 64),
        "case 1 does not rebuild from the seed",
    ),
    "case bound": (
        lambda r: first_case(r).update(bound=1.0),
        "does not rebuild",
    ),
    "padding": (
        lambda r: first_case(r)["launches"]["x_rows"].update(padding_threads=0),
        "does not rebuild",
    ),
    "sample": (
        lambda r: shorten_first_sample(first_point(r)["sides"]["torch_matmul"]),
        "point 0 at",
    ),
    "ratio": (
        lambda r: first_point(r)["naive_to_torch_gflops_ratio"].update(
            matmul_x_rows=2.0
        ),
        "does not recompute from its samples",
    ),
    "disagreement": (
        lambda r: first_point(r).update(torch_disagreement=1.0),
        "must agree with the naive output",
    ),
    "steps": (
        lambda r: first_point(r)["access_steps"]["x_rows"]["a_step_bytes"].update(x=4),
        "does not recompute",
    ),
    "ceiling": (
        lambda r: r["benchmark"]["ceiling"].update(ceiling_index=1),
        "benchmark section does not recompute",
    ),
    "measured flag": (
        lambda r: r.update(measured=True),
        "only the host timer and torch.mps.compile_shader",
    ),
    "boundary": (
        lambda r: r["evidence_boundary"].update(vendor_peak_claimed=True),
        "weakened",
    ),
    "source": (
        lambda r: r["shaders"].update(source=r["shaders"]["source"] + " "),
        "this lab's Metal source",
    ),
    "limits": (
        lambda r: r["shaders"]["kernels"][0]["limits"].update(
            max_threads_per_threadgroup=128
        ),
        "exceeds its pipeline's recorded limit",
    ),
    "environment": (
        lambda r: r["environment"].update(numpy_version="1.26 4"),
        "version or platform token",
    ),
    "planned work": (
        lambda r: r["planned_work"].update(calls=1),
        "planned work does not match",
    ),
    "run": (
        lambda r: r["run"].update(elapsed_ns=1),
        "run time does not cover its samples",
    ),
    "summary": (
        lambda r: r["summary"].update(naive_outputs_replay_exactly=False),
        "summary does not reconcile",
    ),
    "text": (
        lambda r: r["correctness"].update(bound="gamma_K(2**-24) * (|A| @ |B|)"),
        "lab's reference and bound",
    ),
    "field": (lambda r: r.pop("summary"), "fields do not match the schema"),
    "config": (
        lambda r: r["config"]["benchmark"].update(repeats=2),
        "repeats must be in",
    ),
    "config digest": (
        lambda r: r.update(config_sha256="0" * 64),
        "config digest does not match the embedded config",
    ),
    "kind": (lambda r: r.update(kind="matmul_metal_other"), "unexpected report kind"),
    "schema version": (
        lambda r: r.update(schema_version=2),
        "unsupported report schema",
    ),
    "timer": (
        lambda r: r.update(timer="time.monotonic_ns"),
        "measurement method fields were changed",
    ),
    "environment field": (
        lambda r: r["environment"].update(host_name="lab-mac"),
        "environment fields do not match the host record",
    ),
    "timed total": (
        lambda r: r["run"].update(timed_ns=r["run"]["timed_ns"] + 1),
        "run time does not cover its samples",
    ),
    "budget": (
        lambda r: r["run"].update(max_runtime_ns=2 * r["run"]["max_runtime_ns"]),
        "run time does not cover its samples",
    ),
    "elapsed over budget": (
        lambda r: r["run"].update(elapsed_ns=r["run"]["max_runtime_ns"] + 1),
        "run time does not cover its samples",
    ),
    "torch guard": (
        lambda r: first_case(r)["outputs"][2].update(guard_elements=5),
        "torch_matmul on mps is malformed",
    ),
    "torch device": (
        lambda r: first_case(r)["outputs"][2].update(device="cpu"),
        "torch_matmul on mps is malformed",
    ),
    "torch count above the output": (
        overcount_differing,
        "differing elements consistently",
    ),
    "limits above the API": (
        raise_both_thread_limits,
        "outside what the API can report",
    ),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_every_resealed_tamper_is_caught_by_recomputation(name):
    _, report = synthetic_report()
    change, message = TAMPERS[name]

    with pytest.raises(MatmulMetalVerificationError, match=message):
        verify_report(resealed(report, change))


def test_a_measurement_must_come_from_macos():
    report = json.loads(RECORDED_REPORT_JSON)

    tampered = resealed(
        report, lambda r: r["environment"].update(platform_system="Linux")
    )

    with pytest.raises(MatmulMetalVerificationError, match="must come from macOS"):
        verify_report(tampered)


def claim_a_measurement(report):
    report.update(
        measured=True,
        measurement_scope=MEASUREMENT_SCOPE,
        label=MEASUREMENT_LABEL,
        evidence_boundary=dict(MEASUREMENT_BOUNDARY),
    )
    report["summary"]["label"] = MEASUREMENT_LABEL


def test_a_stand_in_timed_by_the_host_timer_is_never_a_measurement():
    config = config_from_mapping(make_config_mapping())
    report = measure_report(
        config, backend=RecordedBackend(), clock=time.perf_counter_ns
    )

    summary = verify_report(report, config=config)

    assert (report["timer"], report["backend"]) == (TIMER, "injected")
    assert (summary["measured"], summary["scope"]) == (False, SYNTHETIC_SCOPE)
    with pytest.raises(
        MatmulMetalVerificationError,
        match="only the host timer and torch.mps.compile_shader",
    ):
        verify_report(resealed(report, claim_a_measurement))


def claim_three_repeats(report):
    """Relabel a four-repeat report as the three-repeat config, as a forger could."""

    config = config_from_mapping(make_config_mapping())
    repeats = config.benchmark.repeats
    report.update(
        config=config.to_mapping(),
        config_sha256=config.fingerprint(),
        planned_work=config.planned_work().to_mapping(),
    )
    points = report["benchmark"]["points"]
    for point in points:
        for side in point["sides"].values():
            side["repeats"] = repeats
    report["summary"]["timed_samples"] = 3 * len(points) * repeats


def test_each_side_must_hold_exactly_the_configured_repeats():
    four = {**make_config_mapping()["benchmark"], "repeats": 4}
    _, report = synthetic_report(benchmark=four)

    tampered = resealed(report, claim_three_repeats)

    with pytest.raises(MatmulMetalVerificationError, match="hold 3 positive integers"):
        verify_report(tampered)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize(("start", "a", "b", "fused", "twice_rounded"), TRAPS)
def test_the_verifier_replay_rounds_once_next_to_a_float32_midpoint(
    start, a, b, fused, twice_rounded, sign
):
    # Seeded operands almost never put a float64 sum on a float32 midpoint,
    # so only these pairs show that rounding to odd stops a second rounding.
    left, right = trap_operands(sign * start, sign * a, b)

    replayed = _replay(left, right)

    assert replayed.dtype == numpy.float32
    assert replayed[0, 0] == numpy.float32(sign * fused)
    assert replayed[0, 0] == exact_fma(sign * a, b, numpy.float32(sign * start))
    widened = numpy.float64(sign * a) * numpy.float64(b) + numpy.float64(sign * start)
    assert numpy.float32(widened) == numpy.float32(sign * twice_rounded)


def lower_a_replay_equal_torch_error(report):
    """Quarter one torch error whose digest says it holds the replay's bits.

    The record's ratio and the summary are rebuilt to match, so only the
    error itself is false.
    """

    cases = report["correctness"]["cases"]
    case = cases[1]
    output = case["outputs"][3]
    assert output["buffer_sha256"] == case["replay_sha256"]
    output["max_relative_error"] /= 4
    output["bound_ratio"] = output["max_relative_error"] / case["bound"]
    summary = report["summary"]
    summary["max_relative_error"]["torch_matmul/cpu"] = max(
        entry["outputs"][3]["max_relative_error"] for entry in cases
    )
    summary["max_bound_ratio"] = max(
        entry["max_relative_error"] / item["bound"]
        for item in cases
        for entry in item["outputs"]
    )


def test_a_torch_record_with_the_replay_digest_must_carry_the_replay_error():
    report = json.loads(RECORDED_REPORT_JSON)

    tampered = resealed(report, lower_a_replay_equal_torch_error)

    with pytest.raises(MatmulMetalVerificationError, match="the replay's error"):
        verify_report(tampered)


def test_an_unsealed_change_fails_the_digest_first():
    _, report = synthetic_report()
    report["summary"]["benchmark_points"] += 1

    with pytest.raises(MatmulMetalVerificationError, match="digest does not match"):
        verify_report(report)
    with pytest.raises(MatmulMetalVerificationError, match="must be a JSON object"):
        verify_report([report])
    with pytest.raises(TypeError, match="MatmulMetalConfig"):
        verify_report(report, config=make_config_mapping())


def test_the_verifier_rebuilds_reports_where_torch_cannot_be_imported(tmp_path):
    _, report = synthetic_report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_metal_verify import verify_report\n"
        "summary = verify_report(json.loads(open(sys.argv[1]).read()))\n"
        "print(summary['measured'], summary['correctness_cases'])\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "False 3\n"
