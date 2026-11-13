import json
import shutil
import sys
from hashlib import sha256

import pytest
from test_serving_recovery_experiment import FIXTURE, experiment_config

import ai_journey.serving_recovery_experiment as experiment
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_experiment import run_serving_recovery_experiment
from ai_journey.serving_recovery_report import (
    build_serving_report,
    seal_serving_report,
    write_serving_report,
)
from ai_journey.serving_recovery_verify import verify_serving_recovery_run


@pytest.fixture(scope="module")
def sealed_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("serving-verify")
    config = base / "config.json"
    config.write_text(json.dumps(experiment_config().to_mapping()))
    run_dir = base / "run"
    result = run_serving_recovery_experiment(
        source_path=FIXTURE, config_path=config, run_dir=run_dir
    )
    write_serving_report(
        run_dir / "report.json", build_serving_report(result, run_dir=run_dir)
    )
    return run_dir


def copy_run(sealed_run, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(sealed_run, target)
    return target


def reseal(run_dir, mutate=None, refresh=()):
    """Rewrite the report as a forger would: edit, refresh digests, reseal."""

    report = json.loads((run_dir / "report.json").read_bytes())
    report.pop("report_sha256")
    if mutate is not None:
        mutate(report)
    for name in refresh:
        digest = sha256((run_dir / name).read_bytes()).hexdigest()
        if name.startswith("journal-"):
            for entry in report["evidence"]["journals"]:
                if entry["file"] == name:
                    entry["file_sha256"] = digest
        else:
            report["evidence"][name.split(".")[0] + "_file_sha256"] = digest
    (run_dir / "report.json").write_bytes(
        canonical_json_bytes(seal_serving_report(report))
    )


def edit_json(path, mutate):
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(canonical_json_bytes(payload))


def test_verifier_recomputes_and_accepts_genuine_evidence(sealed_run):
    receipt = verify_serving_recovery_run(sealed_run)
    report = json.loads((sealed_run / "report.json").read_bytes())

    assert receipt["verified"] is True and receipt["recovered"] is True
    assert receipt["report_sha256"] == report["report_sha256"]
    assert receipt["requests"] == 24
    assert receipt["summary_ms"] == report["summary_ms"]


def stretch_first_ok(payload):
    first = next(a for a in payload["attempts"] if a["outcome"] == "ok")
    first["finished_ns"] += 1


def move_first_port(payload):
    payload["attempts"][0]["port"] = 1


def advance_first_failure(run_dir):
    """Move the first post-kill failure 1 ns earlier, shifting unavailability."""

    injected = json.loads((run_dir / "report.json").read_bytes())["fault"]
    injected_ns = injected["injected_ns"]

    def mutate(payload):
        failures = [
            a
            for a in payload["attempts"]
            if a["outcome"] != "ok" and a["finished_ns"] > injected_ns + 1
        ]
        first = min(failures, key=lambda a: a["finished_ns"])
        first["started_ns"] -= 1
        first["finished_ns"] -= 1

    edit_json(run_dir / "attempts.json", mutate)


def reroute_response(payload):
    first = next(a for a in payload["attempts"] if a["outcome"] == "ok")
    first["response_sha256"] = "0" * 64


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda d: reseal(d, lambda r: r["timing"].update(detection_latency_ns=1)),
            "timing",
        ),
        (
            lambda d: reseal(
                d,
                lambda r: r["reconciliation"]["totals"].update(duplicate_executions=3),
            ),
            "reconciliation",
        ),
        (
            lambda d: reseal(d, lambda r: r.update(evidence_boundary="production")),
            "boundary",
        ),
        (lambda d: reseal(d, lambda r: r.update(recovered=False)), "recovered"),
        (lambda d: reseal(d, lambda r: r["fault"].update(injected_ns=1)), "fault"),
        (
            lambda d: (
                advance_first_failure(d),
                reseal(d, refresh=["attempts.json"]),
            ),
            "timing",
        ),
        (
            lambda d: (
                edit_json(d / "attempts.json", reroute_response),
                reseal(d, refresh=["attempts.json"]),
            ),
            "recomputed output",
        ),
        (
            lambda d: (
                (d / "journal-lifetime-2.jsonl").write_bytes(
                    b"".join(
                        (d / "journal-lifetime-2.jsonl")
                        .read_bytes()
                        .splitlines(True)[1:]
                    )
                ),
                reseal(d, refresh=["journal-lifetime-2.jsonl"]),
            ),
            "sequence|execution",
        ),
        (
            lambda d: edit_json(d / "attempts.json", stretch_first_ok),
            "attempt log digest",
        ),
        (
            lambda d: (
                edit_json(d / "attempts.json", move_first_port),
                reseal(d, refresh=["attempts.json"]),
            ),
            "port",
        ),
        (lambda d: (d / "source.txt").write_bytes(b"zara\n"), "source"),
        (
            lambda d: edit_json(
                d / "trace.json", lambda t: t[1]["payload"].update(port=1)
            ),
            "digest",
        ),
        (lambda d: (d / "report.json").write_bytes(b"{}"), "sealed"),
        (lambda d: (d / "report.json").unlink(), "report"),
    ],
)
def test_verifier_rejects_forged_or_tampered_evidence(
    sealed_run, tmp_path, tamper, message
):
    run_dir = copy_run(sealed_run, tmp_path)
    tamper(run_dir)
    with pytest.raises(ValueError, match=message):
        verify_serving_recovery_run(run_dir)


def as_float(section, name):
    return lambda report: report[section].update({name: float(report[section][name])})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: r["reconciliation"]["totals"].update(failed=False),
            "reconciliation",
        ),
        (as_float("timing", "detection_latency_ns"), "timing"),
        (as_float("fault", "acknowledged_responses"), "fault"),
        (as_float("experiment", "source_bytes"), "source"),
        (lambda r: r["slo"].update(recovered=1), "SLO"),
        (lambda r: r["experiment"]["config"].update(seed=70.0), "config"),
        (
            lambda r: r["evidence"]["journals"][0].update(torn_tail_bytes=False),
            "journal counts",
        ),
    ],
)
def test_verifier_rejects_report_values_that_only_compare_equal(
    sealed_run, tmp_path, mutate, message
):
    run_dir = copy_run(sealed_run, tmp_path)
    reseal(run_dir, mutate)
    with pytest.raises(ValueError, match=message):
        verify_serving_recovery_run(run_dir)


def test_verifier_binds_the_entry_of_a_lifetime_that_never_journaled(
    tmp_path, monkeypatch
):
    real = experiment.server_command

    def command(**options):
        if options["journal_path"].name == "journal-lifetime-2.jsonl":
            return (sys.executable, "-I", "-c", "raise SystemExit(4)")
        return real(**options)

    monkeypatch.setattr(experiment, "server_command", command)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(experiment_config().to_mapping()))
    run_dir = tmp_path / "run"
    result = run_serving_recovery_experiment(
        source_path=FIXTURE, config_path=config, run_dir=run_dir
    )
    write_serving_report(
        run_dir / "report.json", build_serving_report(result, run_dir=run_dir)
    )
    assert [event.kind for event in result.events][4:8] == [
        "restart",
        "exit_observed",
        "restart",
        "ready",
    ]
    assert not (run_dir / "journal-lifetime-2.jsonl").exists()
    assert verify_serving_recovery_run(run_dir)["verified"] is True

    reseal(run_dir, lambda r: r["evidence"]["journals"][1].update(records=3))
    with pytest.raises(ValueError, match="missing journal"):
        verify_serving_recovery_run(run_dir)
