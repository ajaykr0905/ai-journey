import json
import shutil
import sys
from hashlib import sha256

import pytest
from test_serving_recovery_experiment import FIXTURE, experiment_config

import ai_journey.serving_recovery_experiment as experiment
from ai_journey.serving_recovery_client import plan_requests
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_experiment import (
    journal_capacity,
    run_serving_recovery_experiment,
)
from ai_journey.serving_recovery_identity import serving_run_id
from ai_journey.serving_recovery_io import load_attempt_log, load_trace
from ai_journey.serving_recovery_journal import ExecutionRecord, read_execution_journal
from ai_journey.serving_recovery_metrics import measure_serving_recovery
from ai_journey.serving_recovery_model import build_serving_model, completion_body
from ai_journey.serving_recovery_reconcile import reconcile_serving_run
from ai_journey.serving_recovery_report import (
    build_serving_report,
    seal_serving_report,
    summary_milliseconds,
    write_serving_report,
)
from ai_journey.serving_recovery_trace import ServingLifecycleEvent, chain_serving_trace
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


def rechain(run_dir, mutate):
    """Edit trace payloads, re-chain every digest and reseal, as a forger would."""

    events = []
    for record in json.loads((run_dir / "trace.json").read_bytes()):
        mutate(record)
        fields = {
            key: value
            for key, value in record.items()
            if key not in ("previous_sha256", "event_sha256")
        }
        events.append(ServingLifecycleEvent(**fields))
    chained = chain_serving_trace(events)
    (run_dir / "trace.json").write_bytes(canonical_json_bytes(chained))
    head = chained[-1]["event_sha256"]
    reseal(
        run_dir,
        lambda report: report["evidence"].update(trace_head_sha256=head),
        refresh=["trace.json"],
    )


def payload_of(kind, **changes):
    def mutate(record):
        if record["kind"] == kind:
            record["payload"].update(changes)

    return mutate


def test_rechained_traces_with_a_declared_schedule_still_verify(sealed_run, tmp_path):
    faithful = copy_run(sealed_run, tmp_path / "faithful")
    rechain(faithful, lambda record: None)
    assert verify_serving_recovery_run(faithful)["verified"] is True
    # A restart whose spawn failed is untraced but spends the first delay.
    skipped = copy_run(sealed_run, tmp_path / "skipped")
    rechain(skipped, payload_of("restart", backoff_ms=40))
    assert verify_serving_recovery_run(skipped)["verified"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (payload_of("restart", backoff_ms=4321), "backoff schedule"),
        (payload_of("restart", backoff_ms=80), "backoff schedule"),
        (payload_of("recovered", consecutive_successes=1), "traced SLO"),
        (payload_of("recovered", slo_latency_ms=10_000), "traced SLO"),
    ],
)
def test_verifier_binds_the_traced_lifecycle_to_the_controls(
    sealed_run, tmp_path, mutate, message
):
    run_dir = copy_run(sealed_run, tmp_path)
    rechain(run_dir, mutate)
    with pytest.raises(ValueError, match=message):
        verify_serving_recovery_run(run_dir)


def retime(run_dir):
    """Recompute the timing section from edited attempts, as a forger would."""

    config = experiment_config()
    _, events = load_trace(run_dir / "trace.json")
    fault = next(e for e in events if e.kind == "fault")
    exit_event = next(
        e
        for e in events
        if e.kind == "exit_observed" and e.lifetime_id == fault.lifetime_id
    )
    ready = next(e for e in events if e.kind == "ready" and e.sequence > fault.sequence)
    timing = measure_serving_recovery(
        attempts=load_attempt_log(run_dir / "attempts.json"),
        fault_injected_ns=fault.monotonic_ns,
        faulted_lifetime_id=fault.lifetime_id,
        exit_observed_ns=exit_event.monotonic_ns,
        replacement_lifetime_id=ready.lifetime_id,
        replacement_ready_ns=ready.monotonic_ns,
        slo_latency_ms=config.slo_latency_ms,
        slo_consecutive_successes=config.slo_consecutive_successes,
    ).to_mapping()
    reseal(
        run_dir,
        lambda report: report.update(
            timing=timing, summary_ms=summary_milliseconds(timing)
        ),
        refresh=["attempts.json"],
    )


def start_an_acknowledgement_after_its_server_exited(run_dir):
    """Move a pre-fault ack to start after its killed lifetime's exit."""

    _, events = load_trace(run_dir / "trace.json")
    fault = next(e for e in events if e.kind == "fault")
    recovered = next(e for e in events if e.kind == "recovered")
    # After the SLO run and before the final stop, so no other check notices.
    assert recovered.monotonic_ns + 2 < events[-1].monotonic_ns

    def mutate(payload):
        first = next(a for a in payload["attempts"] if a["outcome"] == "ok")
        assert first["lifetime_id"] == fault.lifetime_id
        first["started_ns"] = recovered.monotonic_ns + 1
        first["finished_ns"] = recovered.monotonic_ns + 2

    edit_json(run_dir / "attempts.json", mutate)
    retime(run_dir)


def credit_execution(run_dir, attempt, lifetime_id):
    """Append ``attempt``'s execution to one lifetime's journal and reseal."""

    config = experiment_config()
    model = build_serving_model(FIXTURE.read_bytes())
    plan = plan_requests(
        config,
        run_id=serving_run_id(config, model_sha256=model.model_sha256),
        model=model,
    )
    planned = plan[attempt.request_index]
    report = json.loads((run_dir / "report.json").read_bytes())
    entries = report["evidence"]["journals"]
    entry = next(item for item in entries if item["lifetime_id"] == lifetime_id)
    path = run_dir / entry["file"]
    data = path.read_bytes()
    committed = data[: len(data) - entry["torn_tail_bytes"]]
    record = ExecutionRecord(
        lifetime_id=lifetime_id,
        sequence=entry["records"],
        idempotency_key=attempt.idempotency_key,
        attempt=attempt.attempt,
        request_sha256=planned.request.request_sha256(),
        response_sha256=sha256(completion_body(model, planned.request)).hexdigest(),
        model_sha256=model.model_sha256,
    )
    path.write_bytes(committed + canonical_json_bytes(record.to_mapping()) + b"\n")
    journals = [
        read_execution_journal(
            run_dir / item["file"],
            lifetime_id=item["lifetime_id"],
            max_records=journal_capacity(config),
            allow_torn_tail=True,
        )
        for item in entries
    ]
    reconciliation = reconcile_serving_run(
        plan=plan,
        attempts=load_attempt_log(run_dir / "attempts.json"),
        journals=journals,
        retry_budget=config.retry_budget,
        model_sha256=model.model_sha256,
    ).to_mapping()

    def mutate(report):
        report["reconciliation"] = reconciliation
        for item in report["evidence"]["journals"]:
            if item["lifetime_id"] == lifetime_id:
                item.update(records=item["records"] + 1, torn_tail_bytes=0)

    reseal(run_dir, mutate, refresh=["attempts.json", entry["file"]])


def journal_a_refused_attempt(run_dir):
    """Credit the replacement with executing an attempt whose connect was refused."""

    attempts = load_attempt_log(run_dir / "attempts.json")
    refused = next(a for a in attempts if a.outcome == "connection_refused")
    _, events = load_trace(run_dir / "trace.json")
    credit_execution(run_dir, refused, events[-1].lifetime_id)


def journal_an_attempt_sent_after_the_killed_lifetime_exited(run_dir):
    """Credit the killed lifetime with an attempt sent after its observed exit.

    The last refusal is relabeled a reset, so it counts as connected; only its
    start, after the observed exit, shows the killed server never received it.
    """

    _, events = load_trace(run_dir / "trace.json")
    fault = next(e for e in events if e.kind == "fault")
    exited_ns = next(
        e.monotonic_ns
        for e in events
        if e.kind == "exit_observed" and e.lifetime_id == fault.lifetime_id
    )
    late = max(
        (
            a
            for a in load_attempt_log(run_dir / "attempts.json")
            if a.outcome == "connection_refused"
        ),
        key=lambda a: a.started_ns,
    )
    assert late.started_ns > exited_ns

    def relabel(payload):
        (item,) = (
            a
            for a in payload["attempts"]
            if (a["idempotency_key"], a["attempt"])
            == (late.idempotency_key, late.attempt)
        )
        item.update(outcome="connection_reset", error="ConnectionResetError")

    edit_json(run_dir / "attempts.json", relabel)
    credit_execution(run_dir, late, fault.lifetime_id)


@pytest.mark.parametrize(
    ("forge", "message"),
    [
        (
            start_an_acknowledgement_after_its_server_exited,
            "acknowledgement started after",
        ),
        (journal_a_refused_attempt, "never connected"),
        (
            journal_an_attempt_sent_after_the_killed_lifetime_exited,
            "journal executed an attempt sent after",
        ),
    ],
)
def test_verifier_rejects_attempts_the_lifecycle_makes_impossible(
    sealed_run, tmp_path, forge, message
):
    run_dir = copy_run(sealed_run, tmp_path)
    forge(run_dir)
    with pytest.raises(ValueError, match=message):
        verify_serving_recovery_run(run_dir)
