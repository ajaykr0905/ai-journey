import json
from pathlib import Path

import pytest
from test_telemetry_traces_serving import sealed_serving_run
from test_telemetry_traces_worker import sealed_worker_run

from ai_journey.telemetry_traces_config import load_telemetry_config
from ai_journey.telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    canonical_sha256,
    sha256_hex,
)
from ai_journey.telemetry_traces_report import (
    ENCODING,
    EXPORT_FILE,
    REPORT_FILE,
    REPORT_KIND,
    build_trace_export,
    build_trace_export_report,
    load_trace_export_report,
    seal_report,
    write_trace_export,
)
from ai_journey.telemetry_traces_validate import validate_export_request
from ai_journey.trace_context import parse_traceparent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_telemetry_config(ROOT / "config" / "day-73-telemetry-traces.json")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-report")
    return sealed_serving_run(base), sealed_worker_run(base)


@pytest.fixture(scope="module")
def export(runs):
    serving, worker = runs
    return build_trace_export(CONFIG, serving_run_dir=serving, worker_run_dir=worker)


def test_both_labs_export_as_one_validated_request_with_a_resource_each(export, runs):
    request = json.loads(export.data)

    assert export.data == artifact_bytes(request) and request == export.request
    assert validate_export_request(request) == export.summary
    assert export.summary.resources == 2
    serving, worker = export.sources
    assert (serving["lab"], worker["lab"]) == ("serving-recovery", "worker-recovery")
    for source, run_dir in zip(export.sources, runs):
        report = json.loads((run_dir / "report.json").read_bytes())
        assert source["report_sha256"] == report["report_sha256"]
        assert source["evidence_boundary"] == report["evidence_boundary"]
        assert source["time_mapping"]["anchor_unix_nano"] == CONFIG.anchor_unix_nano
        context = parse_traceparent(source["traceparent"])
        assert (context.trace_id, context.span_id) == (
            source["trace_id"],
            source["root_span_id"],
        )
    assert sorted(s["trace_id"] for s in export.sources) == list(
        export.summary.trace_ids
    )
    assert serving["spans"] + worker["spans"] == export.summary.spans
    assert str(runs[0]) not in json.dumps(export.sources)


def test_the_sealed_report_binds_bytes_sources_config_and_boundary(export):
    report = build_trace_export_report(CONFIG, export)

    unsigned = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == canonical_sha256(unsigned)
    assert (report["kind"], report["schema_version"]) == (REPORT_KIND, 1)
    assert report["telemetry_scope"] == "local-cpu-lab"
    assert (
        "not production, GPU, CUDA, distributed or multi-host"
        in (report["evidence_boundary"])
    )
    assert report["config"] == CONFIG.to_mapping()
    assert report["config_sha256"] == CONFIG.fingerprint()
    assert report["encoding"] == ENCODING
    assert report["export"] == {
        "file": EXPORT_FILE,
        "bytes": len(export.data),
        "sha256": sha256_hex(export.data),
        **export.summary.to_mapping(),
    }
    assert report["sources"] == list(export.sources)


def test_exports_publish_into_a_new_directory_outside_the_runs(export, runs, tmp_path):
    report = build_trace_export_report(CONFIG, export)
    output = tmp_path / "telemetry"

    write_trace_export(output, export, report, sources=runs)

    assert sorted(path.name for path in output.iterdir()) == [REPORT_FILE, EXPORT_FILE]
    assert (output / EXPORT_FILE).read_bytes() == export.data
    assert load_trace_export_report(output / REPORT_FILE) == report
    with pytest.raises(TelemetryError, match="already exist"):
        write_trace_export(output, export, report, sources=runs)
    with pytest.raises(TelemetryError, match="inside a source"):
        write_trace_export(runs[0] / "telemetry", export, report, sources=runs)
    assert not (runs[0] / "telemetry").exists()
    forged = seal_report(
        {k: v for k, v in report.items() if k != "report_sha256"}
        | {"export": report["export"] | {"sha256": "0" * 64}}
    )
    with pytest.raises(TelemetryError, match="does not seal these export bytes"):
        write_trace_export(tmp_path / "forged", export, forged, sources=runs)


def test_reports_load_only_when_canonical_sealed_and_of_this_kind(export, tmp_path):
    report = build_trace_export_report(CONFIG, export)
    path = tmp_path / "report.json"
    path.write_bytes(artifact_bytes(report | {"telemetry_scope": "production"}))
    with pytest.raises(TelemetryError, match="seal does not match"):
        load_trace_export_report(path)
    path.write_bytes(artifact_bytes(seal_report({"kind": "other"})))
    with pytest.raises(TelemetryError, match="kind"):
        load_trace_export_report(path)
    path.write_bytes(json.dumps(report).encode())
    with pytest.raises(TelemetryError, match="canonical"):
        load_trace_export_report(path)


def test_a_single_lab_exports_alone_and_an_empty_export_is_refused(runs):
    worker_only = build_trace_export(CONFIG, worker_run_dir=runs[1])

    assert [source["lab"] for source in worker_only.sources] == ["worker-recovery"]
    assert worker_only.summary.spans == 6
    with pytest.raises(TelemetryError, match="at least one source"):
        build_trace_export(CONFIG)


def test_an_unverifiable_source_stops_the_export(runs, tmp_path):
    missing = tmp_path / "missing-run"
    with pytest.raises(TelemetryVerificationError, match="evidence was rejected"):
        build_trace_export(CONFIG, serving_run_dir=runs[0], worker_run_dir=missing)
