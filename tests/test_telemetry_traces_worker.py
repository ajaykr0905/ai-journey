import json
import shutil
from hashlib import sha256

import pytest
from test_recovery_config import valid_config

import ai_journey.telemetry_traces_worker as worker
from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_report import build_recovery_report
from ai_journey.recovery_report_io import write_recovery_report
from ai_journey.recovery_trace import RecoveryEvent, chain_recovery_event
from ai_journey.telemetry_traces_io import TelemetryVerificationError
from ai_journey.telemetry_traces_otlp import encode_export_request
from ai_journey.telemetry_traces_validate import validate_export_request
from ai_journey.telemetry_traces_worker import (
    load_verified_worker_evidence,
    worker_resource,
)

ANCHOR = 1_794_787_200_000_000_000


def sealed_worker_run(base):
    """Kill and resume one real worker process and seal its report."""

    source = base / "worker-source.txt"
    source.write_bytes(b"public recovery fixture")
    config = base / "worker-config.json"
    config.write_text(json.dumps(valid_config().to_mapping()))
    run_dir = base / "worker-run"
    result = run_recovery_experiment(
        source_path=source, config_path=config, run_dir=run_dir
    )
    write_recovery_report(run_dir / "report.json", build_recovery_report(result))
    return run_dir


@pytest.fixture(scope="module")
def worker_run(tmp_path_factory):
    return sealed_worker_run(tmp_path_factory.mktemp("telemetry-worker"))


@pytest.fixture(scope="module")
def converted(worker_run):
    evidence = load_verified_worker_evidence(worker_run)
    return evidence, worker_resource(evidence, anchor_unix_nano=ANCHOR)


def test_a_verified_worker_run_becomes_attempt_and_phase_spans(converted):
    evidence, resource = converted
    report, records = evidence.report, evidence.records
    origin = records[0]["monotonic_ns"]
    root, failed, resumed, *phases = resource.spans

    assert root.parent_span_id is None and root.name == "worker-recovery run"
    assert root.end_unix_nano - root.start_unix_nano == (
        records[-1]["monotonic_ns"] - origin
    )
    assert root.attributes["ai_journey.items.lost"] == 3
    assert root.attributes["ai_journey.items.omitted"] == 0
    assert (failed.name, resumed.name) == ("worker attempt 1", "worker attempt 2")
    assert failed.parent_span_id == resumed.parent_span_id == root.span_id
    assert (
        failed.status == "error" and failed.attributes["error.type"] == "process_kill"
    )
    assert (
        failed.attributes["process.exit.code"] == report["attempts"]["failed_exit_code"]
    )
    assert resumed.status == "unset" and resumed.attributes["process.exit.code"] == 0
    assert resumed.attributes["ai_journey.attempt.start_items"] == 8
    assert [event.name for event in failed.events + resumed.events] == [
        f"ai_journey.lifecycle.{record['kind']}" for record in records
    ]
    for event in failed.events + resumed.events:
        record = records[event.attributes["ai_journey.event.sequence"]]
        assert event.attributes["ai_journey.event.sha256"] == record["event_sha256"]
        assert event.time_unix_nano == ANCHOR + record["monotonic_ns"] - origin
        assert (
            event.attributes["ai_journey.work.completed_items"]
            == (record["completed_items"])
        )
    assert [phase.name for phase in phases] == [
        "failure detection",
        "worker restart",
        "recovery",
    ]
    for phase in phases:
        field = phase.attributes["ai_journey.phase.source"].removeprefix("timing.")
        duration_ns = phase.end_unix_nano - phase.start_unix_nano
        assert duration_ns / 1_000_000 == report["timing"][field]
    assert resource.attributes["service.name"] == "ai-journey-worker-recovery-lab"
    assert resource.attributes["deployment.environment.name"] == "local-cpu-lab"
    assert (
        resource.attributes["ai_journey.evidence.trace_head_sha256"]
        == (report["trace_sha256"])
    )


def test_worker_spans_encode_to_a_valid_export(converted):
    _, resource = converted
    summary = validate_export_request(encode_export_request([resource]))

    assert summary.spans == 6 and summary.events == 5
    assert summary.spans_by_kind == {"internal": 6}
    assert summary.error_spans == 1


def copy_run(worker_run, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(worker_run, target)
    return target


def digest(payload):
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rewrite(run_dir, records):
    """Rechain records, then rebind and reseal the report as a forger would."""

    chained = []
    for record in records:
        event = RecoveryEvent(
            **{
                k: v
                for k, v in record.items()
                if k not in ("previous_sha256", "event_sha256")
            }
        )
        chained.append(
            chain_recovery_event(
                event, chained[-1]["event_sha256"] if chained else None
            )
        )
    (run_dir / "trace.json").write_bytes(
        json.dumps(chained, sort_keys=True, separators=(",", ":")).encode()
    )
    report = json.loads((run_dir / "report.json").read_bytes())
    report.pop("report_sha256")
    report["trace_sha256"] = chained[-1]["event_sha256"]
    report["report_sha256"] = digest(report)
    (run_dir / "report.json").write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )


def records_of(run_dir):
    return json.loads((run_dir / "trace.json").read_bytes())


def reseal(run_dir, mutate):
    """Edit the report and reseal it exactly as build_recovery_report does."""

    report = json.loads((run_dir / "report.json").read_bytes())
    report.pop("report_sha256")
    mutate(report)
    report["report_sha256"] = digest(report)
    (run_dir / "report.json").write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )


def test_a_report_resealed_with_another_boundary_is_never_converted(
    worker_run, tmp_path
):
    genuine = json.loads((worker_run / "report.json").read_bytes())
    assert genuine["evidence_boundary"] == worker.WORKER_EVIDENCE_BOUNDARY
    run_dir = copy_run(worker_run, tmp_path)
    # verify_recovery_report checks the seal but never the boundary text.
    claim = "Production GPU fleet monitoring across 512 hosts."
    reseal(run_dir, lambda r: r.update(evidence_boundary=claim))

    with pytest.raises(TelemetryVerificationError, match="evidence boundary"):
        load_verified_worker_evidence(run_dir)


def test_a_trace_too_short_for_the_lab_verifier_is_rejected_not_a_crash(
    worker_run, tmp_path
):
    run_dir = copy_run(worker_run, tmp_path)
    started, fault = records_of(run_dir)[:2]
    # A chain that ends early, at the fault time, passes the trace verifier.
    early_end = fault | {"kind": "completed", "payload": {}}
    rewrite(run_dir, [started, early_end])

    with pytest.raises(TelemetryVerificationError, match="evidence was rejected"):
        load_verified_worker_evidence(run_dir)


def test_only_the_experiments_five_record_shape_is_converted(worker_run, tmp_path):
    run_dir = copy_run(worker_run, tmp_path)
    records = records_of(run_dir)
    rewrite(run_dir, records + [records[4] | {"sequence": 5}])

    with pytest.raises(TelemetryVerificationError, match="five-record shape"):
        load_verified_worker_evidence(run_dir)


def as_float(section, name):
    def mutate(report):
        report[section][name] = float(report[section][name])

    return mutate


# verify_recovery_report compares by value, so each of these still verifies.
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: r["loss_assessment"].update(omitted_items=False),
            "exact integers",
        ),
        (lambda r: r["attempts"].update(resumed_exit_code=False), "exact integers"),
        (lambda r: r["fault"].update(checkpoint_interval=True), "exact integers"),
        (
            lambda r: (
                as_float("experiment", "work_items")(r),
                as_float("loss_assessment", "final_items")(r),
            ),
            "exact integers",
        ),
        (
            lambda r: r["experiment"].update(run_id="day-70-other-0123456789abcdef"),
            "differ from the verified report",
        ),
        (
            lambda r: r["attempts"].update(parent_checkpoint_sha256="1" * 64),
            "differ from the verified report",
        ),
    ],
)
def test_resealed_reports_must_keep_exact_types_and_trace_bindings(
    worker_run, tmp_path, mutate, message
):
    run_dir = copy_run(worker_run, tmp_path)
    reseal(run_dir, mutate)

    with pytest.raises(TelemetryVerificationError, match=message):
        load_verified_worker_evidence(run_dir)


def set_payload(sequence, **payload):
    def mutate(records):
        records[sequence]["payload"].update(payload)

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_payload(2, exit_code=-9.0), "payload differs"),
        (set_payload(0, exit_code=0), "payload differs"),
        (set_payload(0, workload_sha256="1" * 64), "differ from the verified report"),
        (set_payload(1, mechanism="sigterm"), "differ from the verified report"),
        (set_payload(4, recovery_time_ms=0.5), "differ from the verified report"),
        (
            lambda records: records[3].update(completed_items=4),
            "differ from the verified report",
        ),
    ],
)
def test_rechained_traces_must_carry_the_experiments_payloads(
    worker_run, tmp_path, mutate, message
):
    run_dir = copy_run(worker_run, tmp_path)
    records = records_of(run_dir)
    mutate(records)
    rewrite(run_dir, records)

    with pytest.raises(TelemetryVerificationError, match=message):
        load_verified_worker_evidence(run_dir)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: (d / "trace.json").unlink(),
        lambda d: (d / "report.json").write_bytes(
            (d / "report.json")
            .read_bytes()
            .replace(b'"omitted_items":0', b'"omitted_items":1')
        ),
        lambda d: (d / "report.json").write_bytes(
            json.dumps(json.loads((d / "report.json").read_bytes()), indent=1).encode()
        ),
        lambda d: (d / "report.json").write_bytes(
            (d / "report.json").read_bytes().replace(b"{", b'{"schema_version":1,', 1)
        ),
    ],
)
def test_rejected_or_ambiguous_worker_evidence_is_never_converted(
    worker_run, tmp_path, tamper
):
    run_dir = copy_run(worker_run, tmp_path)
    tamper(run_dir)
    with pytest.raises(TelemetryVerificationError, match="evidence was rejected"):
        load_verified_worker_evidence(run_dir)


def test_a_trace_swapped_after_verification_is_refused(
    worker_run, tmp_path, monkeypatch
):
    run_dir = copy_run(worker_run, tmp_path)
    verify = worker.verify_recovery_report

    def verify_then_swap(report, *, trace_path):
        verify(report, trace_path=trace_path)
        records = records_of(run_dir)
        records[4]["payload"]["accumulator_sha256"] = "0" * 64
        chained = []
        for record in records:
            fields = {
                k: v
                for k, v in record.items()
                if k not in ("previous_sha256", "event_sha256")
            }
            chained.append(
                chain_recovery_event(
                    RecoveryEvent(**fields),
                    chained[-1]["event_sha256"] if chained else None,
                )
            )
        trace_path.write_bytes(
            json.dumps(chained, sort_keys=True, separators=(",", ":")).encode()
        )

    monkeypatch.setattr(worker, "verify_recovery_report", verify_then_swap)
    with pytest.raises(TelemetryVerificationError, match="trace head differs"):
        load_verified_worker_evidence(run_dir)
