import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_metal_config import config_from_mapping
from ai_journey.matmul_metal_report import measure_report
from ai_journey.matmul_metal_verify import MatmulMetalVerificationError, verify_report
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from test_matmul_metal_config import make_config_mapping
from test_matmul_metal_recorded import RecordedBackend

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


def first_case(report):
    return report["correctness"]["cases"][1]


def first_point(report):
    return report["benchmark"]["points"][0]


def flip_differing(output):
    """Claim other bits where the digest says none, or none where it says some."""

    output["differing_elements"] = int(output["differing_elements"] == 0)


def shorten_first_sample(side):
    side["samples_ns"][0] = 7


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
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_every_resealed_tamper_is_caught_by_recomputation(name):
    _, report = synthetic_report()
    change, message = TAMPERS[name]

    with pytest.raises(MatmulMetalVerificationError, match=message):
        verify_report(resealed(report, change))


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
