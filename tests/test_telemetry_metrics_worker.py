import json
import shutil
from hashlib import sha256

import pytest
from test_recovery_config import valid_config as worker_config

from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_report import build_recovery_report
from ai_journey.recovery_report_io import write_recovery_report
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.telemetry_metrics_exposition import encode_exposition, parse_exposition
from ai_journey.telemetry_metrics_model import TelemetryMetricsError, metric_set
from ai_journey.telemetry_metrics_worker import (
    load_verified_worker_evidence,
    worker_recovery_families,
)


def make_worker_run(base):
    """Run one real worker kill-and-resume experiment and write its report."""

    source = base / "worker-source.txt"
    source.write_bytes(b"public recovery fixture")
    config = base / "worker-config.json"
    config.write_text(json.dumps(worker_config().to_mapping()))
    run_dir = base / "worker-run"
    result = run_recovery_experiment(
        source_path=source, config_path=config, run_dir=run_dir
    )
    write_recovery_report(run_dir / "report.json", build_recovery_report(result))
    return run_dir


@pytest.fixture(scope="module")
def worker_run(tmp_path_factory):
    return make_worker_run(tmp_path_factory.mktemp("telemetry-worker"))


def rewrite(run_dir, mutate):
    report = json.loads((run_dir / "report.json").read_bytes())
    mutate(report)
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(canonical_json_bytes(unsigned)).hexdigest()
    (run_dir / "report.json").write_bytes(canonical_json_bytes(report))


def test_families_restate_the_verified_loss_and_exact_timing(worker_run):
    evidence = load_verified_worker_evidence(worker_run)
    report = evidence.report
    loss, timing = report["loss_assessment"], report["timing"]
    families = {
        family.name.removeprefix("ai_journey_worker_recovery_"): family
        for family in worker_recovery_families(evidence)
    }

    def value(name, **labels):
        family = families[name]
        (series,) = [
            item
            for item in family.series
            if all(item.label_map().get(k) == v for k, v in labels.items())
        ]
        assert series.label_map()["scope"] == "local_cpu_lab"
        return series.value

    assert value("completed_items_total") == loss["final_items"] == 20
    assert value("lost_items_total") == loss["lost_uncheckpointed_items"] == 3
    assert value("replayed_items_total") == loss["replayed_items"]
    assert value("omitted_items_total") == value("duplicated_items_total") == 0
    assert value("items_at_fault") == loss["fault_observed_items"]
    assert value("durable_items_at_fault") == loss["durable_items_at_fault"]
    assert value("detection_seconds") == (
        (timing["failure_detected_ns"] - timing["fault_injected_ns"]) / 1e9
    )
    assert value("restart_delay_seconds") == (
        (timing["restart_started_ns"] - timing["failure_detected_ns"]) / 1e9
    )
    assert value("recovery_seconds") == (
        (timing["recovery_completed_ns"] - timing["fault_injected_ns"]) / 1e9
    )
    assert value("recovery_seconds") == pytest.approx(timing["recovery_time_ms"] / 1e3)
    assert value("attempt_exit_code", attempt="failed") != 0
    assert value("attempt_exit_code", attempt="resumed") == 0
    info = families["run_info"].series[0].label_map()
    assert info["run_id"] == report["experiment"]["run_id"]
    assert info["report_sha256"] == report["report_sha256"]
    assert info["trace_sha256"] == report["trace_sha256"]
    for family in families.values():
        assert family.help.endswith("from one verified worker recovery run.")


def test_worker_families_encode_and_parse_back(worker_run):
    metrics = metric_set(
        worker_recovery_families(load_verified_worker_evidence(worker_run))
    )
    text = encode_exposition(metrics)

    assert parse_exposition(text) == metrics
    assert 'ai_journey_worker_recovery_lost_items_total{scope="local_cpu_lab"} 3\n' in (
        text
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (
            lambda d: rewrite(
                d, lambda r: r["loss_assessment"].update(replayed_items=2)
            ),
            "does not reconcile",
        ),
        (
            lambda d: rewrite(d, lambda r: r.update(trace_sha256="f" * 64)),
            "trace identity",
        ),
        (
            lambda d: rewrite(
                d,
                lambda r: r["timing"].update(
                    fault_injected_ns=r["timing"]["fault_injected_ns"] - 1
                ),
            ),
            "durations|timing",
        ),
        (
            lambda d: (d / "report.json").write_text(
                json.dumps(json.loads((d / "report.json").read_bytes()), indent=1)
            ),
            "canonical",
        ),
        (lambda d: (d / "report.json").write_text('{"a":1,"a":1}'), "duplicate"),
        (lambda d: (d / "trace.json").unlink(), "trace"),
        (lambda d: (d / "report.json").unlink(), "readable regular"),
    ],
)
def test_rejected_worker_evidence_yields_no_metrics(
    worker_run, tmp_path, damage, message
):
    copy = tmp_path / "copy"
    shutil.copytree(worker_run, copy)
    damage(copy)

    with pytest.raises(TelemetryMetricsError, match="evidence was rejected") as caught:
        load_verified_worker_evidence(copy)
    assert caught.match(message)


def test_values_the_worker_verifier_compares_loosely_must_be_integers(
    worker_run, tmp_path
):
    copy = tmp_path / "copy"
    shutil.copytree(worker_run, copy)
    rewrite(copy, lambda r: r["loss_assessment"].update(final_items=20.0))

    with pytest.raises(TelemetryMetricsError, match="must be integers"):
        load_verified_worker_evidence(copy)
    with pytest.raises(TypeError):
        load_verified_worker_evidence(str(worker_run))
    with pytest.raises(TypeError):
        worker_recovery_families({"report": {}})
