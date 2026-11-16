import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_telemetry_traces_serving import sealed_serving_run
from test_telemetry_traces_worker import sealed_worker_run

from ai_journey.telemetry_traces_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    build_parser,
    main,
)
from ai_journey.telemetry_traces_io import artifact_bytes, seal_report

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-73-telemetry-traces.json"


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(arg) for arg in argv])
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-cli")
    return sealed_serving_run(base), sealed_worker_run(base)


@pytest.fixture(scope="module")
def exported(runs, tmp_path_factory):
    serving, worker = runs
    output = tmp_path_factory.mktemp("telemetry-cli-out") / "traces"
    result = run(
        "export",
        "--config",
        PINNED,
        "--serving-run-dir",
        serving,
        "--worker-run-dir",
        worker,
        "--output-dir",
        output,
    )
    return output, result


def test_export_writes_a_verified_otlp_export_and_prints_its_receipt(exported):
    output, (code, out, err) = exported

    assert (code, err) == (EXIT_OK, "")
    receipt = json.loads(out)
    report = json.loads((output / "report.json").read_bytes())
    assert receipt["verified"] is True and receipt["telemetry_scope"] == "local-cpu-lab"
    assert receipt["report_sha256"] == report["report_sha256"]
    assert receipt["spans"] == report["export"]["spans"]
    assert [trace["lab"] for trace in receipt["traces"]] == [
        "serving-recovery",
        "worker-recovery",
    ]
    assert sorted(path.name for path in output.iterdir()) == [
        "report.json",
        "traces.otlp.json",
    ]


def test_verify_recomputes_the_export_and_binds_the_pinned_config(
    exported, runs, tmp_path
):
    output, (_, out, _) = exported
    serving, worker = runs
    arguments = ["--serving-run-dir", serving, "--worker-run-dir", worker]

    code, verified, _ = run("verify", "--report", output / "report.json", *arguments)
    assert code == EXIT_OK and json.loads(verified) == json.loads(out)
    other = tmp_path / "other.json"
    other.write_text(PINNED.read_text().replace('"1m"', '"5m"'))
    code, _, err = run(
        "verify", "--report", output / "report.json", "--config", other, *arguments
    )
    assert code == EXIT_REJECTED and "another config" in err
    code, _, err = run(
        "verify", "--report", output / "report.json", "--serving-run-dir", serving
    )
    assert code == EXIT_INVALID and "exactly the exported labs" in err


def test_tampered_exports_and_unverifiable_sources_are_rejected(
    exported, runs, tmp_path
):
    output, _ = exported
    serving, worker = runs
    copy = tmp_path / "traces"
    shutil.copytree(output, copy)
    request = json.loads((copy / "traces.otlp.json").read_bytes())
    request["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] = "forged"
    (copy / "traces.otlp.json").write_bytes(artifact_bytes(request))
    arguments = ["--serving-run-dir", serving, "--worker-run-dir", worker]

    code, _, err = run("verify", "--report", copy / "report.json", *arguments)
    assert code == EXIT_REJECTED and "sealed digest" in err

    broken = tmp_path / "worker-copy"
    shutil.copytree(worker, broken)
    (broken / "trace.json").unlink()
    destination = tmp_path / "never-written"
    code, out, err = run(
        "export",
        "--config",
        PINNED,
        "--worker-run-dir",
        broken,
        "--output-dir",
        destination,
    )
    assert (code, out) == (EXIT_REJECTED, "")
    assert "evidence was rejected" in err and not destination.exists()


def test_refused_outputs_and_inputs_are_invalid_and_write_nothing(runs, tmp_path):
    serving, worker = runs
    taken = tmp_path / "taken"
    taken.mkdir()
    cases = [
        (
            "export",
            "--config",
            PINNED,
            "--worker-run-dir",
            worker,
            "--output-dir",
            taken,
        ),
        (
            "export",
            "--config",
            PINNED,
            "--worker-run-dir",
            worker,
            "--output-dir",
            worker / "telemetry",
        ),
        ("export", "--config", PINNED, "--output-dir", tmp_path / "no-sources"),
        (
            "export",
            "--config",
            tmp_path / "absent.json",
            "--worker-run-dir",
            worker,
            "--output-dir",
            tmp_path / "x",
        ),
        ("dashboard", "--config", PINNED, "--output-dir", taken),
        ("verify", "--report", tmp_path / "absent.json"),
    ]
    for argv in cases:
        code, out, err = run(*argv)
        assert (code, out) == (EXIT_INVALID, ""), argv
        assert "failed:" in err
    assert list(taken.iterdir()) == [] and not (worker / "telemetry").exists()
    assert not (tmp_path / "no-sources").exists() and not (tmp_path / "x").exists()


def test_dashboard_writes_verified_json_that_verify_accepts(tmp_path, runs):
    output = tmp_path / "dashboard"
    code, out, err = run("dashboard", "--config", PINNED, "--output-dir", output)

    assert (code, err) == (EXIT_OK, "")
    receipt = json.loads(out)
    assert receipt["uid"] == "ai-journey-day-73-recovery"
    assert (receipt["schema_version"], receipt["panels"]) == (39, 7)
    assert sorted(path.name for path in output.iterdir()) == [
        "dashboard.json",
        "report.json",
    ]
    code, verified, _ = run(
        "verify", "--report", output / "report.json", "--config", PINNED
    )
    assert code == EXIT_OK and json.loads(verified) == receipt
    code, _, err = run(
        "verify", "--report", output / "report.json", "--worker-run-dir", runs[1]
    )
    assert code == EXIT_INVALID and "only to trace export" in err


def test_reports_of_unknown_kind_are_rejected(tmp_path):
    report = tmp_path / "report.json"
    report.write_bytes(artifact_bytes(seal_report({"kind": "other"})))
    code, _, err = run("verify", "--report", report)
    assert code == EXIT_REJECTED and "neither a trace export nor a dashboard" in err
    report.write_bytes(b'{"kind": 1, "kind": 2}')
    code, _, err = run("verify", "--report", report)
    assert code == EXIT_REJECTED and "duplicate JSON key" in err


def test_usage_errors_exit_two_and_the_module_runs_as_a_script(tmp_path):
    with pytest.raises(SystemExit) as usage:
        build_parser().parse_args(["export", "--config", str(PINNED)])
    assert usage.value.code == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args(["export", "--conf", str(PINNED)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_journey.telemetry_traces_cli",
            "dashboard",
            "--config",
            str(PINNED),
            "--output-dir",
            str(tmp_path / "script"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"] is True


def test_the_cli_imports_neither_torch_nor_numpy():
    probe = (
        "import sys, ai_journey.telemetry_traces_cli; "
        "print([m for m in ('torch', 'numpy', 'matplotlib') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.stdout.strip() == "[]"
