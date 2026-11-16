import json
import shutil

import pytest
from test_telemetry_metrics_report import RELAXED_CONFIG
from test_telemetry_metrics_serving import make_serving_run
from test_telemetry_metrics_worker import make_worker_run

import ai_journey.telemetry_metrics_report as report_module
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.telemetry_metrics_model import TelemetryMetricsError
from ai_journey.telemetry_metrics_report import (
    build_metrics_export,
    seal_metrics_report,
    write_metrics_export,
)
from ai_journey.telemetry_metrics_verify import verify_metrics_export


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-verify")
    serving, worker = make_serving_run(base), make_worker_run(base)
    output = base / "export"
    write_metrics_export(
        output,
        build_metrics_export(
            RELAXED_CONFIG, serving_run_dir=serving, worker_run_dir=worker
        ),
    )
    return output, serving, worker


def verify(output, serving, worker):
    return verify_metrics_export(output, serving_run_dir=serving, worker_run_dir=worker)


def copy_export(sealed, tmp_path):
    output, serving, worker = sealed
    target = tmp_path / "copy"
    shutil.copytree(output, target)
    return target, serving, worker


def reseal(output, mutate):
    report = json.loads((output / "report.json").read_bytes())
    report.pop("report_sha256")
    mutate(report)
    (output / "report.json").write_bytes(
        canonical_json_bytes(seal_metrics_report(report))
    )


def edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_verifier_recomputes_and_accepts_a_genuine_export(sealed):
    output, serving, worker = sealed
    receipt = verify(output, serving, worker)
    report = json.loads((output / "report.json").read_bytes())

    assert receipt["verified"] is True
    assert receipt["report_sha256"] == report["report_sha256"]
    assert receipt["scope"] == "local_cpu_lab"
    assert (receipt["families"], receipt["recording_rules"]) == (28, 3)
    assert receipt["breached_objectives"] == receipt["firing_alerts"] == []


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda d: edit(
                d / "metrics.prom",
                'attempts_total{outcome="ok",scope="local_cpu_lab"} ',
                'attempts_total{outcome="ok",scope="local_cpu_lab"} 1',
            ),
            "samples differ",
        ),
        (
            lambda d: edit(
                d / "metrics.prom",
                'ai_journey_worker_recovery_lost_items_total{scope="local_cpu_lab"} 3',
                'ai_journey_worker_recovery_lost_items_total{scope="local_cpu_lab"} 3.0',
            ),
            "canonical encoding",
        ),
        (
            lambda d: edit(d / "metrics.prom", "# HELP", "# lab note\n# HELP"),
            "comment lines",
        ),
        (lambda d: (d / "metrics.prom").write_bytes(b"\xff\n"), "not UTF-8"),
        (lambda d: edit(d / "rules.yml", " > 300'", " > 299'"), "differ from"),
        (lambda d: edit(d / "rules.yml", "groups:\n", "groups: \n"), "trailing"),
        (lambda d: (d / "rules.yml").unlink(), "readable regular"),
        (lambda d: reseal(d, lambda r: r.update(scope="production")), "scope"),
        (
            lambda d: reseal(d, lambda r: r.update(evidence_boundary="production")),
            "boundary",
        ),
        (lambda d: reseal(d, lambda r: r.update(extra=1)), "fields"),
        (lambda d: reseal(d, lambda r: r.update(schema_version=True)), "schema"),
        (
            lambda d: reseal(d, lambda r: r["objectives"][0].update(breached=True)),
            "report objectives",
        ),
        (
            lambda d: reseal(
                d,
                lambda r: r["firing_alerts"].append(
                    {"alert": "WorkerRecoverySlow", "labels": {}, "value": "1"}
                ),
            ),
            "report firing_alerts",
        ),
        (
            lambda d: reseal(d, lambda r: r["exposition"].update(samples=1)),
            "report exposition",
        ),
        (
            lambda d: reseal(d, lambda r: r["evidence"]["serving"].update(requests=25)),
            "report evidence",
        ),
        (
            lambda d: reseal(d, lambda r: r["config"].update(experiment_name="x")),
            "config",
        ),
        (
            lambda d: (d / "config.json").write_bytes(
                json.dumps(RELAXED_CONFIG.to_mapping(), indent=1).encode()
            ),
            "not canonical",
        ),
        (
            lambda d: (d / "config.json").write_text(
                '{"schema_version":1,"schema_version":1}'
            ),
            "duplicate JSON key",
        ),
        (
            lambda d: (d / "report.json").write_bytes(
                (d / "report.json")
                .read_bytes()
                .replace(b'"local_cpu_lab"', b'"lab"', 1)
            ),
            "seal",
        ),
    ],
)
def test_verifier_rejects_every_tampered_export(sealed, tmp_path, tamper, message):
    output, serving, worker = copy_export(sealed, tmp_path)
    tamper(output)

    with pytest.raises(TelemetryMetricsError, match=message):
        verify(output, serving, worker)


def test_consistently_reconfigured_exports_fail_on_the_rules(sealed, tmp_path):
    output, serving, worker = copy_export(sealed, tmp_path)
    changed = RELAXED_CONFIG.to_mapping()
    changed["worker_objectives"]["recovery_ms"] = 299_999
    (output / "config.json").write_bytes(canonical_json_bytes(changed))
    reseal(output, lambda r: r.update(config=changed))

    with pytest.raises(TelemetryMetricsError, match="config"):
        verify(output, serving, worker)
    reseal(
        output,
        lambda r: r.update(config_sha256=report_module.canonical_sha256(changed)),
    )
    with pytest.raises(TelemetryMetricsError, match="stored rules differ"):
        verify(output, serving, worker)


def test_other_or_damaged_evidence_does_not_verify_an_export(sealed, tmp_path):
    output, serving, worker = sealed
    other = make_serving_run(tmp_path)

    with pytest.raises(TelemetryMetricsError):
        verify(output, other, worker)
    damaged = tmp_path / "damaged"
    shutil.copytree(serving, damaged)
    (damaged / "trace.json").write_bytes(b"[]")
    with pytest.raises(TelemetryMetricsError, match="serving-recovery evidence"):
        verify(output, damaged, worker)
    with pytest.raises(TelemetryMetricsError, match="worker-recovery evidence"):
        verify(output, serving, serving)
    with pytest.raises(TelemetryMetricsError, match="output_dir is invalid"):
        verify(tmp_path / "missing", serving, worker)
    with pytest.raises(TypeError):
        verify(str(output), serving, worker)


def test_cross_checks_catch_an_adapter_that_mislabels_its_samples(
    sealed, tmp_path, monkeypatch
):
    _, serving, worker = sealed
    real = report_module.serving_recovery_families

    def mislabeled(evidence, *, latency_buckets_us):
        families = real(evidence, latency_buckets_us=latency_buckets_us)
        swapped = []
        for family in families:
            if family.name.endswith("_lost_response_executions_total"):
                family = family.__class__(
                    family.name,
                    family.kind,
                    family.help,
                    tuple(
                        series.__class__(series.labels, series.value + 1)
                        for series in family.series
                    ),
                )
            swapped.append(family)
        return tuple(swapped)

    monkeypatch.setattr(report_module, "serving_recovery_families", mislabeled)
    output = tmp_path / "mislabeled"
    write_metrics_export(
        output,
        build_metrics_export(
            RELAXED_CONFIG, serving_run_dir=serving, worker_run_dir=worker
        ),
    )
    with pytest.raises(TelemetryMetricsError, match="lost_response_executions"):
        verify(output, serving, worker)
