import json
from pathlib import Path

import pytest

import ai_journey.telemetry_dashboard_report as dashboard_report
from ai_journey.telemetry_dashboard_grafana import DASHBOARD_BOUNDARY, build_dashboard
from ai_journey.telemetry_dashboard_metrics import METRICS
from ai_journey.telemetry_dashboard_report import (
    DASHBOARD_FILE,
    REPORT_FILE,
    REPORT_KIND,
    build_dashboard_artifact,
    build_dashboard_report,
    verify_dashboard_report,
    write_dashboard_artifact,
)
from ai_journey.telemetry_traces_config import load_telemetry_config
from ai_journey.telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    seal_report,
    sha256_hex,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-73-telemetry-traces.json"
CONFIG = load_telemetry_config(PINNED)


@pytest.fixture
def published(tmp_path):
    artifact = build_dashboard_artifact(CONFIG)
    output = tmp_path / "dashboard"
    write_dashboard_artifact(output, artifact, build_dashboard_report(CONFIG, artifact))
    return output


def test_the_report_seals_dashboard_bytes_catalog_scope_and_boundary():
    artifact = build_dashboard_artifact(CONFIG)
    report = build_dashboard_report(CONFIG, artifact)

    assert artifact.data == artifact_bytes(build_dashboard(CONFIG))
    assert (
        report["kind"] == REPORT_KIND and report["telemetry_scope"] == "local-cpu-lab"
    )
    assert report["evidence_boundary"] == DASHBOARD_BOUNDARY
    assert report["grafana"] == {
        "schema_version": 39,
        "uid": "ai-journey-day-73-recovery",
        "title": "AI Journey local CPU lab: recovery telemetry",
        "datasource": "prometheus through the ${datasource} variable",
        "imported_into_grafana": False,
    }
    assert [metric["name"] for metric in report["metrics"]] == [m.name for m in METRICS]
    assert report["dashboard"] == {
        "file": DASHBOARD_FILE,
        "bytes": len(artifact.data),
        "sha256": sha256_hex(artifact.data),
        **artifact.summary.to_mapping(),
    }


def test_a_published_dashboard_verifies_against_the_pinned_config(published):
    receipt = verify_dashboard_report(published / REPORT_FILE, config_path=PINNED)
    report = json.loads((published / REPORT_FILE).read_bytes())

    assert receipt == {
        "verified": True,
        "kind": REPORT_KIND,
        "telemetry_scope": "local-cpu-lab",
        "report_sha256": report["report_sha256"],
        "dashboard_sha256": sha256_hex((published / DASHBOARD_FILE).read_bytes()),
        "uid": "ai-journey-day-73-recovery",
        "schema_version": 39,
        "panels": 7,
        "targets": 12,
    }
    assert sorted(path.name for path in published.iterdir()) == [
        DASHBOARD_FILE,
        REPORT_FILE,
    ]
    artifact = build_dashboard_artifact(CONFIG)
    report = build_dashboard_report(CONFIG, artifact)
    with pytest.raises(TelemetryError, match="already exist"):
        write_dashboard_artifact(published, artifact, report)


def edit_report(directory, mutate, *, reseal=True):
    report = json.loads((directory / REPORT_FILE).read_bytes())
    report.pop("report_sha256")
    mutate(report)
    sealed = seal_report(report) if reseal else report | {"report_sha256": "0" * 64}
    (directory / REPORT_FILE).write_bytes(artifact_bytes(sealed))


def edit_dashboard(directory, mutate):
    dashboard = json.loads((directory / DASHBOARD_FILE).read_bytes())
    mutate(dashboard)
    data = artifact_bytes(dashboard)
    (directory / DASHBOARD_FILE).write_bytes(data)
    edit_report(
        directory,
        lambda r: r["dashboard"].update(bytes=len(data), sha256=sha256_hex(data)),
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda d: edit_report(d, lambda r: None, reseal=False), "seal does not match"),
        (
            lambda d: edit_report(d, lambda r: r.update(evidence_boundary="fleet")),
            "boundary changed",
        ),
        (lambda d: edit_report(d, lambda r: r.pop("config")), "fields"),
        (
            lambda d: edit_report(d, lambda r: r["metrics"].pop()),
            "metric catalog differs",
        ),
        (
            lambda d: edit_report(
                d, lambda r: r["grafana"].update(imported_into_grafana=True)
            ),
            "grafana section changed",
        ),
        (
            lambda d: edit_report(d, lambda r: r["dashboard"].update(panels=6)),
            "summary does not match",
        ),
        (
            lambda d: edit_dashboard(
                d, lambda b: b["panels"][1].update(title="Renamed panel")
            ),
            "does not rebuild",
        ),
        (
            lambda d: edit_dashboard(
                d, lambda b: b["panels"][2]["gridPos"].update(x=11)
            ),
            "overlaps",
        ),
        (
            lambda d: edit_dashboard(
                d,
                lambda b: b["panels"][1]["targets"][0].update(
                    expr="sum(ai_journey_serving_attempt_total)"
                ),
            ),
            "unknown metric",
        ),
        (
            lambda d: (d / DASHBOARD_FILE).write_bytes(
                (d / DASHBOARD_FILE).read_bytes().replace(b"stat", b"gauge", 1)
            ),
            "sealed digest",
        ),
        (lambda d: (d / DASHBOARD_FILE).unlink(), "dashboard must be a readable"),
    ],
)
def test_tampered_dashboards_and_reports_are_rejected(published, tamper, message):
    tamper(published)
    with pytest.raises(TelemetryVerificationError, match=message):
        verify_dashboard_report(published / REPORT_FILE)


def test_a_report_from_another_config_fails_the_pinned_binding(published, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(PINNED.read_text().replace('"refresh": "1m"', '"refresh": "5m"'))

    with pytest.raises(TelemetryVerificationError, match="another config"):
        verify_dashboard_report(published / REPORT_FILE, config_path=other)
    with pytest.raises(TelemetryError, match="readable") as unreadable:
        verify_dashboard_report(published / "missing.json")
    assert not isinstance(unreadable.value, TelemetryVerificationError)


def test_a_dashboard_missing_a_catalog_metric_is_not_built(monkeypatch):
    def without_worker_panels(config):
        dashboard = build_dashboard(config)
        dashboard["panels"] = dashboard["panels"][:-2]
        return dashboard

    monkeypatch.setattr(dashboard_report, "build_dashboard", without_worker_panels)
    with pytest.raises(TelemetryError, match="does not chart ai_journey_worker"):
        build_dashboard_artifact(CONFIG)
