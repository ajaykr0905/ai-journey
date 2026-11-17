import json
import shutil
import signal
from dataclasses import replace

import pytest
from test_serving_recovery_experiment import FIXTURE, experiment_config

import ai_journey.telemetry_traces_serving as serving
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_experiment import run_serving_recovery_experiment
from ai_journey.serving_recovery_report import (
    REPORT_FILE,
    build_serving_report,
    seal_serving_report,
    write_serving_report,
)
from ai_journey.telemetry_traces_io import TelemetryVerificationError
from ai_journey.telemetry_traces_otlp import encode_export_request
from ai_journey.telemetry_traces_serving import (
    load_verified_serving_evidence,
    serving_resource,
)
from ai_journey.telemetry_traces_validate import validate_export_request

ANCHOR = 1_794_787_200_000_000_000


def sealed_serving_run(base):
    """Run one real loopback kill-and-recover experiment and seal its report."""

    config = base / "serving-config.json"
    config.write_text(json.dumps(experiment_config().to_mapping()))
    run_dir = base / "serving-run"
    result = run_serving_recovery_experiment(
        source_path=FIXTURE, config_path=config, run_dir=run_dir
    )
    write_serving_report(
        run_dir / REPORT_FILE, build_serving_report(result, run_dir=run_dir)
    )
    return run_dir


@pytest.fixture(scope="module")
def serving_run(tmp_path_factory):
    return sealed_serving_run(tmp_path_factory.mktemp("telemetry-serving"))


@pytest.fixture(scope="module")
def converted(serving_run):
    evidence = load_verified_serving_evidence(serving_run)
    return evidence, serving_resource(evidence, anchor_unix_nano=ANCHOR)


def named(resource, prefix):
    return [span for span in resource.spans if span.name.startswith(prefix)]


def test_a_verified_run_becomes_one_labeled_trace_of_its_lifecycle(converted):
    evidence, resource = converted
    report, records = evidence.report, evidence.records
    origin = records[0]["monotonic_ns"]

    (root,) = [span for span in resource.spans if span.parent_span_id is None]
    assert root.name == "serving-recovery run" and root.status == "unset"
    assert (root.start_unix_nano, root.end_unix_nano) == (
        ANCHOR,
        ANCHOR + records[-1]["monotonic_ns"] - origin,
    )
    assert root.attributes["ai_journey.run.id"] == report["experiment"]["run_id"]
    assert len(resource.trace_ids) == 1
    first, second = named(resource, "server lifetime")
    assert (first.name, second.name) == ("server lifetime 1", "server lifetime 2")
    assert first.status == "error" and first.attributes["error.type"] == "SIGKILL"
    assert first.status_message == "SIGKILL injected after 6 acknowledged responses"
    assert first.attributes["process.exit.code"] == -signal.SIGKILL
    assert second.status == "unset" and second.attributes["process.exit.code"] == 0
    assert (
        second.attributes["ai_journey.lifetime.parent_id"] == records[0]["lifetime_id"]
    )
    assert second.attributes["server.address"] == "127.0.0.1"

    lifecycle = [(event, span) for span in resource.spans for event in span.events]
    assert sorted(
        e.attributes["ai_journey.event.sequence"] for e, _ in lifecycle
    ) == list(range(len(records)))
    for event, span in lifecycle:
        record = records[event.attributes["ai_journey.event.sequence"]]
        assert event.attributes["ai_journey.event.sha256"] == record["event_sha256"]
        assert event.name == f"ai_journey.lifecycle.{record['kind']}"
        assert event.time_unix_nano == ANCHOR + record["monotonic_ns"] - origin
        assert span.attributes["ai_journey.lifetime.id"] == record["lifetime_id"]
    (fault,) = [e for e, _ in lifecycle if e.name == "ai_journey.lifecycle.fault"]
    assert fault.attributes["ai_journey.fault.signal"] == "SIGKILL"

    attributes = resource.attributes
    assert attributes["deployment.environment.name"] == "local-cpu-lab"
    assert attributes["service.name"] == "ai-journey-serving-recovery-lab"
    assert attributes["ai_journey.evidence.report_sha256"] == report["report_sha256"]
    assert attributes["ai_journey.evidence.lab_boundary"] == report["evidence_boundary"]
    assert attributes["ai_journey.clock.origin_monotonic_ns"] == origin
    assert attributes["ai_journey.clock.anchor_unix_nano"] == ANCHOR


def test_derived_phases_reproduce_the_sealed_recovery_durations(converted):
    evidence, resource = converted
    timing = evidence.report["timing"]
    phases = [s for s in resource.spans if s.attributes.get("ai_journey.span.derived")]

    assert [span.name for span in phases] == [
        "fault detection",
        "server restart",
        "client unavailability",
        "SLO recovery",
    ]
    origin = evidence.records[0]["monotonic_ns"]
    for span in phases:
        field = span.attributes["ai_journey.phase.source"].removeprefix("timing.")
        assert span.end_unix_nano - span.start_unix_nano == timing[field]
    assert phases[0].start_unix_nano == ANCHOR + timing["fault_injected_ns"] - origin


def test_attempts_are_client_spans_with_http_conventions(converted):
    evidence, resource = converted
    requests = {span.span_id: span for span in named(resource, "serving request")}
    clients = [span for span in resource.spans if span.kind == "client"]
    attempts = {(a.idempotency_key, a.attempt): a for a in evidence.attempts}

    assert len(requests) == evidence.report["reconciliation"]["totals"]["requests"]
    assert len(clients) == len(attempts) > len(requests)
    for span in clients:
        request = requests[span.parent_span_id]
        key = request.attributes["ai_journey.request.idempotency_key"]
        attempt = attempts.pop((key, span.attributes["ai_journey.attempt.number"]))
        assert span.name == "POST /generate"
        assert span.end_unix_nano - span.start_unix_nano == attempt.latency_ns
        assert span.attributes["http.request.method"] == "POST"
        assert span.attributes["server.port"] == attempt.port
        assert (
            span.attributes["url.full"] == f"http://127.0.0.1:{attempt.port}/generate"
        )
        assert span.attributes.get("http.request.resend_count") == (
            None if attempt.attempt == 1 else attempt.attempt - 1
        )
        if attempt.outcome == "ok":
            assert span.status == "unset" and "error.type" not in span.attributes
            assert span.attributes["http.response.status_code"] == 200
            assert span.attributes["ai_journey.lifetime.id"] == attempt.lifetime_id
        else:
            assert span.status == "error"
            assert span.attributes["error.type"] == attempt.outcome
            assert span.attributes["ai_journey.attempt.error"] == attempt.error
    assert not attempts
    retried = [s for s in clients if "http.request.resend_count" in s.attributes]
    assert retried and any(s.status == "unset" for s in retried)


def test_converted_runs_encode_to_a_valid_export_request(converted):
    evidence, resource = converted
    summary = validate_export_request(encode_export_request([resource]))

    report = evidence.report
    totals = report["reconciliation"]["totals"]
    lifetimes = len(report["experiment"]["lifetimes"])
    phases = sum(
        report["timing"][field] is not None for _, _, _, field in serving.PHASES
    )
    assert summary.spans == len(resource.spans)
    assert summary.spans_by_kind == {
        "client": totals["attempts"],
        "internal": 1 + lifetimes + phases + totals["requests"],
    }
    assert summary.events == len(evidence.records)
    ok = report["reconciliation"]["attempt_outcomes"]["ok"]
    assert summary.error_spans == 1 + totals["attempts"] - ok


def test_conversion_is_deterministic_and_the_anchor_only_moves_absolute_times(
    converted,
):
    evidence, resource = converted
    shifted = serving_resource(evidence, anchor_unix_nano=ANCHOR + 1000)

    assert serving_resource(evidence, anchor_unix_nano=ANCHOR) == resource
    assert [s.context for s in shifted.spans] == [s.context for s in resource.spans]
    assert [s.start_unix_nano - 1000 for s in shifted.spans] == [
        s.start_unix_nano for s in resource.spans
    ]


def copy_run(serving_run, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(serving_run, target)
    return target


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: (d / "trace.json").unlink(),
        lambda d: (d / "report.json").write_bytes(b"{}"),
        lambda d: (d / "attempts.json").write_bytes(
            (d / "attempts.json")
            .read_bytes()
            .replace(b'"attempt":1', b'"attempt":2', 1)
        ),
    ],
)
def test_evidence_the_lab_verifier_rejects_is_never_converted(
    serving_run, tmp_path, tamper
):
    run_dir = copy_run(serving_run, tmp_path)
    tamper(run_dir)
    with pytest.raises(TelemetryVerificationError, match="evidence was rejected"):
        load_verified_serving_evidence(run_dir)


def reseal_report(run_dir):
    report = json.loads((run_dir / "report.json").read_bytes())
    report.pop("report_sha256")
    report["evidence_boundary"] = "production fleet"
    (run_dir / "report.json").write_bytes(
        canonical_json_bytes(seal_serving_report(report))
    )


@pytest.mark.parametrize(
    ("swap", "message"),
    [
        (lambda d: (d / "trace.json").write_bytes(b"[]"), "trace.json differs"),
        (
            lambda d: (d / "attempts.json").write_bytes(
                (d / "attempts.json").read_bytes() + b" "
            ),
            "attempts.json differs",
        ),
        (reseal_report, "report differs"),
    ],
)
def test_evidence_swapped_after_verification_is_refused(
    serving_run, tmp_path, monkeypatch, swap, message
):
    run_dir = copy_run(serving_run, tmp_path)
    verify = serving.verify_serving_recovery_run

    def verify_then_swap(path):
        receipt = verify(path)
        swap(path)
        return receipt

    monkeypatch.setattr(serving, "verify_serving_recovery_run", verify_then_swap)
    with pytest.raises(TelemetryVerificationError, match=message):
        load_verified_serving_evidence(run_dir)


def test_sealed_reconciliation_must_cover_every_attempt(converted):
    evidence, _ = converted
    report = json.loads(json.dumps(evidence.report))
    report["reconciliation"]["requests"].pop()
    with pytest.raises(TelemetryVerificationError, match="outside the sealed"):
        serving_resource(replace(evidence, report=report), anchor_unix_nano=ANCHOR)
