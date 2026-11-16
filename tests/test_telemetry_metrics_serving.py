import json
import shutil
from dataclasses import replace

import pytest
from test_serving_recovery_experiment import FIXTURE, experiment_config
from test_telemetry_metrics_config import valid_config

import ai_journey.telemetry_metrics_serving as serving
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_experiment import run_serving_recovery_experiment
from ai_journey.serving_recovery_identity import lifetime_number
from ai_journey.serving_recovery_report import (
    build_serving_report,
    write_serving_report,
)
from ai_journey.telemetry_metrics_exposition import encode_exposition, parse_exposition
from ai_journey.telemetry_metrics_model import TelemetryMetricsError, metric_set
from ai_journey.telemetry_metrics_serving import (
    FAILED_OUTCOMES,
    PHASES,
    load_verified_serving_evidence,
    serving_recovery_families,
)

BUCKETS = valid_config().latency_buckets_us


def make_serving_run(base):
    """Run one real loopback kill-and-recover experiment and seal its report."""

    config = base / "serving-config.json"
    config.write_text(json.dumps(experiment_config().to_mapping()))
    run_dir = base / "serving-run"
    result = run_serving_recovery_experiment(
        source_path=FIXTURE, config_path=config, run_dir=run_dir
    )
    write_serving_report(
        run_dir / "report.json", build_serving_report(result, run_dir=run_dir)
    )
    return run_dir


@pytest.fixture(scope="module")
def serving_run(tmp_path_factory):
    return make_serving_run(tmp_path_factory.mktemp("telemetry-serving"))


@pytest.fixture(scope="module")
def evidence(serving_run):
    return load_verified_serving_evidence(serving_run)


def values(family):
    return {
        tuple(value for name, value in series.labels if name != "scope"): series.value
        for series in family.series
    }


def test_families_restate_the_verified_reconciliation_and_timing(evidence):
    report = evidence.report
    families = {
        family.name.removeprefix("ai_journey_serving_recovery_"): family
        for family in serving_recovery_families(evidence, latency_buckets_us=BUCKETS)
    }
    reconciliation, timing = report["reconciliation"], report["timing"]
    totals = reconciliation["totals"]

    assert evidence.receipt["report_sha256"] == report["report_sha256"]
    assert values(families["attempts_total"]) == {
        (name,): float(count)
        for name, count in reconciliation["attempt_outcomes"].items()
    }
    assert values(families["requests_total"]) == {
        (name,): float(totals[name])
        for name in ("first_try_success", "success_after_retry", "failed_after_budget")
    }
    assert values(families["executions_total"]) == {
        (str(lifetime_number(lifetime)),): float(count)
        for lifetime, count in reconciliation["executions_by_lifetime"].items()
    }
    for name in (
        "duplicate_executions",
        "lost_response_executions",
        "retried_attempts",
    ):
        assert values(families[name + "_total"]) == {(): float(totals[name])}
    assert values(families["late_acknowledgements_total"]) == {
        (): float(timing["late_acknowledgements"])
    }
    assert values(families["restarts_total"]) == {(): 1.0}
    for name in ("detection_latency", "restart_latency", "unavailability"):
        assert values(families[name + "_seconds"]) == {(): timing[name + "_ns"] / 1e9}
    assert values(families["time_to_slo_recovery_seconds"]) == {
        (): timing["time_to_slo_recovery_ns"] / 1e9
    }
    assert values(families["recovered"]) == {(): 1.0}
    assert values(families["run_info"]) == {
        (report["report_sha256"], report["experiment"]["run_id"]): 1.0
    }
    for family in families.values():
        assert family.help.endswith("from one verified serving recovery run.")
        assert all(
            series.label_map()["scope"] == "local_cpu_lab" for series in family.series
        )


def test_latency_histograms_partition_attempts_by_phase_and_outcome(evidence):
    families = serving_recovery_families(evidence, latency_buckets_us=BUCKETS)
    by_name = {family.name: family for family in families}
    success = by_name["ai_journey_serving_recovery_success_latency_seconds"]
    failed = by_name["ai_journey_serving_recovery_failed_attempt_duration_seconds"]
    timing = evidence.report["timing"]
    attempts = evidence.attempts

    assert success.buckets == tuple(bound / 1e6 for bound in BUCKETS)
    counts = {series.label_map()["phase"]: series.value for series in success.series}
    assert set(counts) == set(PHASES)
    assert counts["pre_fault"].count == timing["pre_fault_latency"]["samples"]
    assert counts["post_fault"].count == timing["post_fault_latency"]["samples"]
    assert counts["late_acknowledgement"].count == timing["late_acknowledgements"]
    oks = [a for a in attempts if a.outcome == "ok"]
    assert sum(value.count for value in counts.values()) == len(oks)
    assert sum(value.sum for value in counts.values()) == pytest.approx(
        sum(a.latency_ns for a in oks) / 1e9
    )
    post = counts["post_fault"]
    within = sum(
        a.latency_ns <= BUCKETS[-1] * 1000
        for a in oks
        if lifetime_number(a.lifetime_id) > 1
    )
    assert post.cumulative_counts[len(BUCKETS) - 1] == within
    failures = {series.label_map()["outcome"]: series.value for series in failed.series}
    assert tuple(sorted(failures)) == tuple(sorted(FAILED_OUTCOMES))
    for outcome, value in failures.items():
        assert value.count == sum(a.outcome == outcome for a in attempts)


def test_serving_families_encode_and_parse_back_deterministically(evidence):
    first = metric_set(serving_recovery_families(evidence, latency_buckets_us=BUCKETS))
    second = metric_set(serving_recovery_families(evidence, latency_buckets_us=BUCKETS))
    text = encode_exposition(first)

    assert text == encode_exposition(second)
    assert parse_exposition(text) == first
    assert 'ai_journey_serving_recovery_recovered{scope="local_cpu_lab"} 1\n' in text


def test_rejected_or_missing_evidence_yields_no_metrics(serving_run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(serving_run, copy)
    payload = json.loads((copy / "attempts.json").read_bytes())
    payload["attempts"][0]["finished_ns"] += 1
    (copy / "attempts.json").write_bytes(canonical_json_bytes(payload))

    with pytest.raises(TelemetryMetricsError, match="evidence was rejected"):
        load_verified_serving_evidence(copy)
    (copy / "report.json").unlink()
    with pytest.raises(TelemetryMetricsError, match="evidence was rejected"):
        load_verified_serving_evidence(copy)
    with pytest.raises(TypeError):
        load_verified_serving_evidence(str(serving_run))


def test_files_replaced_after_verification_are_refused(serving_run, monkeypatch):
    receipt = serving.verify_serving_recovery_run(serving_run)
    monkeypatch.setattr(
        serving,
        "verify_serving_recovery_run",
        lambda run_dir: receipt | {"report_sha256": "0" * 64},
    )
    with pytest.raises(TelemetryMetricsError, match="report changed"):
        load_verified_serving_evidence(serving_run)

    monkeypatch.setattr(serving, "verify_serving_recovery_run", lambda run_dir: receipt)
    real = serving.load_attempt_log

    def shifted(path):
        first, *rest = real(path)
        return (replace(first, finished_ns=first.finished_ns + 1), *rest)

    monkeypatch.setattr(serving, "load_attempt_log", shifted)
    with pytest.raises(TelemetryMetricsError, match="attempt log changed"):
        load_verified_serving_evidence(serving_run)


def test_phase_counts_must_match_the_verified_timing(evidence):
    report = json.loads(canonical_json_bytes(evidence.report))
    report["timing"]["late_acknowledgements"] += 1
    forged = replace(evidence, report=report)

    with pytest.raises(TelemetryMetricsError, match="latency phases"):
        serving_recovery_families(forged, latency_buckets_us=BUCKETS)
    with pytest.raises(TypeError):
        serving_recovery_families(report, latency_buckets_us=BUCKETS)
