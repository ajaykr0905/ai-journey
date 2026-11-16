import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_telemetry_traces_serving import sealed_serving_run
from test_telemetry_traces_worker import sealed_worker_run

import ai_journey.telemetry_traces_serving as serving
from ai_journey.telemetry_traces_config import (
    config_from_mapping,
    load_telemetry_config,
)
from ai_journey.telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    sha256_hex,
)
from ai_journey.telemetry_traces_report import (
    EXPORT_FILE,
    REPORT_FILE,
    build_trace_export,
    build_trace_export_report,
    seal_report,
    write_trace_export,
)
from ai_journey.telemetry_traces_verify import verify_trace_export

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_telemetry_config(ROOT / "config" / "day-73-telemetry-traces.json")


def export_to(output, serving_run, worker_run):
    export = build_trace_export(
        CONFIG, serving_run_dir=serving_run, worker_run_dir=worker_run
    )
    report = build_trace_export_report(CONFIG, export)
    write_trace_export(output, export, report, sources=(serving_run, worker_run))
    return output


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-verify")
    runs = sealed_serving_run(base), sealed_worker_run(base)
    return export_to(base / "telemetry", *runs), runs


@pytest.fixture
def copied(sealed, tmp_path):
    exported, runs = sealed
    target = tmp_path / "telemetry"
    shutil.copytree(exported, target)
    return target, runs


def verify(exported, runs):
    serving_run, worker_run = runs
    return verify_trace_export(
        exported / REPORT_FILE, serving_run_dir=serving_run, worker_run_dir=worker_run
    )


def test_a_genuine_export_recomputes_from_its_sources(sealed):
    exported, runs = sealed
    receipt = verify(exported, runs)
    report = json.loads((exported / REPORT_FILE).read_bytes())

    assert receipt["verified"] is True
    assert receipt["telemetry_scope"] == "local-cpu-lab"
    assert receipt["report_sha256"] == report["report_sha256"]
    assert receipt["export_sha256"] == sha256_hex((exported / EXPORT_FILE).read_bytes())
    assert receipt["spans"] == report["export"]["spans"]
    assert [trace["lab"] for trace in receipt["traces"]] == [
        "serving-recovery",
        "worker-recovery",
    ]
    assert all(t["traceparent"].endswith("-01") for t in receipt["traces"])


def edit_report(exported, mutate, *, reseal=True):
    report = json.loads((exported / REPORT_FILE).read_bytes())
    report.pop("report_sha256")
    mutate(report)
    if reseal:
        report = seal_report(report)
    else:
        report["report_sha256"] = "0" * 64
    (exported / REPORT_FILE).write_bytes(artifact_bytes(report))


def edit_export(exported, mutate, *, refresh=True):
    request = json.loads((exported / EXPORT_FILE).read_bytes())
    mutate(request)
    data = artifact_bytes(request)
    (exported / EXPORT_FILE).write_bytes(data)
    if refresh:
        edit_report(
            exported,
            lambda r: r["export"].update(bytes=len(data), sha256=sha256_hex(data)),
        )


def rename_first_span(request):
    request["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] = "forged"


def shift_anchor(report):
    report["config"]["time_mapping"].update(
        anchor_unix_nano=1_794_787_201_000_000_000, anchor_utc="2026-11-16T00:00:01Z"
    )
    report["config_sha256"] = config_from_mapping(report["config"]).fingerprint()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda d: edit_report(d, lambda r: None, reseal=False), "seal does not match"),
        (
            lambda d: edit_report(d, lambda r: r.update(evidence_boundary="prod")),
            "boundary changed",
        ),
        (
            lambda d: edit_report(d, lambda r: r.update(telemetry_scope="production")),
            "scope changed",
        ),
        (lambda d: edit_report(d, lambda r: r.update(extra=1)), "fields"),
        (lambda d: edit_report(d, shift_anchor), "does not recompute"),
        (
            lambda d: edit_report(
                d, lambda r: r["sources"][0].update(traceparent="00-" + "1" * 49)
            ),
            "sources do not recompute",
        ),
        (
            lambda d: edit_report(d, lambda r: r["export"].update(spans=1)),
            "summary does not match",
        ),
        (lambda d: edit_export(d, rename_first_span), "does not recompute"),
        (
            lambda d: edit_export(d, rename_first_span, refresh=False),
            "sealed digest",
        ),
        (lambda d: (d / EXPORT_FILE).unlink(), "export must be a readable"),
        (
            lambda d: (d / REPORT_FILE).write_bytes(
                (d / REPORT_FILE).read_bytes().rstrip(b"\n")
            ),
            "canonical",
        ),
    ],
)
def test_forged_or_tampered_exports_are_rejected(copied, tamper, message):
    exported, runs = copied
    tamper(exported)
    with pytest.raises(TelemetryVerificationError, match=message):
        verify(exported, runs)


def test_a_pinned_config_must_be_the_one_the_export_embeds(copied, tmp_path):
    exported, (serving_run, worker_run) = copied
    pinned = ROOT / "config" / "day-73-telemetry-traces.json"
    other = tmp_path / "other.json"
    other.write_text(pinned.read_text().replace('"now-6h"', '"now-1h"'))
    runs = dict(serving_run_dir=serving_run, worker_run_dir=worker_run)

    verify_trace_export(exported / REPORT_FILE, config_path=pinned, **runs)
    with pytest.raises(TelemetryVerificationError, match="another config"):
        verify_trace_export(exported / REPORT_FILE, config_path=other, **runs)


def test_source_evidence_that_no_longer_verifies_rejects_the_export(copied, tmp_path):
    exported, (serving_run, worker_run) = copied
    changed = tmp_path / "serving-copy"
    shutil.copytree(serving_run, changed)
    (changed / "attempts.json").write_bytes(
        (changed / "attempts.json").read_bytes() + b"\n"
    )
    with pytest.raises(TelemetryVerificationError, match="evidence was rejected"):
        verify(exported, (changed, worker_run))


def test_verify_needs_exactly_the_exported_labs_and_a_readable_report(copied):
    exported, (serving_run, worker_run) = copied
    with pytest.raises(TelemetryError, match="exactly the exported labs") as missing:
        verify_trace_export(exported / REPORT_FILE, serving_run_dir=serving_run)
    assert not isinstance(missing.value, TelemetryVerificationError)
    with pytest.raises(TelemetryError, match="readable") as unreadable:
        verify_trace_export(
            exported / "absent.json",
            serving_run_dir=serving_run,
            worker_run_dir=worker_run,
        )
    assert not isinstance(unreadable.value, TelemetryVerificationError)


def shift_lifecycle_event(original):
    def shifted(event, record, at):
        converted = original(event, record, at)
        if event.kind != "fault":
            return converted
        return replace(converted, time_unix_nano=converted.time_unix_nano + 1)

    return shifted


def hide_attempt_errors(original):
    def hidden(attempt, parent, at):
        return replace(
            original(attempt, parent, at), status="unset", status_message=None
        )

    return hidden


@pytest.mark.parametrize(
    ("name", "patch", "message"),
    [
        ("_lifecycle_event", shift_lifecycle_event, "does not map its trace record"),
        ("_attempt_span", hide_attempt_errors, "errors differ from the failed"),
    ],
)
def test_a_self_consistent_encoder_bug_is_caught_against_raw_evidence(
    sealed, tmp_path, monkeypatch, name, patch, message
):
    _, runs = sealed
    monkeypatch.setattr(serving, name, patch(getattr(serving, name)))
    # Export and rebuild share the bug, so only the evidence mapping can catch it.
    exported = export_to(tmp_path / "buggy", *runs)
    with pytest.raises(TelemetryVerificationError, match=message):
        verify(exported, runs)
