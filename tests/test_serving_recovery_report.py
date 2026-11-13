import json
from hashlib import sha256

import pytest
from test_serving_recovery_experiment import FIXTURE, experiment_config

from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_experiment import run_serving_recovery_experiment
from ai_journey.serving_recovery_report import (
    EVIDENCE_BOUNDARY,
    build_serving_report,
    load_serving_report,
    seal_serving_report,
    write_serving_report,
)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    base = tmp_path_factory.mktemp("serving-report")
    config = base / "config.json"
    config.write_text(json.dumps(experiment_config().to_mapping()))
    run_dir = base / "run"
    result = run_serving_recovery_experiment(
        source_path=FIXTURE, config_path=config, run_dir=run_dir
    )
    return result, run_dir, config


def test_report_binds_identities_raw_evidence_and_the_claim_boundary(measured):
    result, run_dir, _ = measured
    report = build_serving_report(result, run_dir=run_dir)

    assert report["kind"] == "ai-journey-serving-recovery-report"
    assert report["experiment"]["config_sha256"] == result.config.fingerprint()
    assert report["experiment"]["model_sha256"] == result.model_sha256
    assert len(report["experiment"]["lifetimes"]) == 2
    evidence = report["evidence"]
    for name in ("attempts", "trace", "config"):
        digest = sha256((run_dir / evidence[f"{name}_file"]).read_bytes()).hexdigest()
        assert evidence[f"{name}_file_sha256"] == digest
    assert [j["records"] for j in evidence["journals"]] == [
        len(journal.records) for journal in result.journals
    ]
    assert report["fault"]["acknowledged_responses"] == 6
    assert report["reconciliation"]["totals"] == result.reconciliation.totals
    assert report["timing"]["detection_latency_ns"] == (
        result.timing.detection_latency_ns
    )
    assert report["summary_ms"]["restart_latency_ms"] == round(
        result.timing.restart_latency_ns / 1e6, 3
    )
    assert report["recovered"] is True and report["slo"]["recovered"] is True
    assert report["evidence_boundary"] == EVIDENCE_BOUNDARY
    for phrase in ("one CPU host", "127.0.0.1", "not", "GPU", "production"):
        assert phrase in EVIDENCE_BOUNDARY
    unsigned = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == sha256(canonical_json_bytes(unsigned)).hexdigest()


def test_report_contains_no_host_paths(measured):
    result, run_dir, _ = measured
    encoded = canonical_json_bytes(build_serving_report(result, run_dir=run_dir))
    assert str(run_dir).encode() not in encoded
    assert str(run_dir.resolve()).encode() not in encoded
    for marker in (b"/Users/", b"/home/", b"/private/", b"/tmp/"):
        assert marker not in encoded


def test_reports_publish_once_and_reject_tampering(measured, tmp_path):
    result, run_dir, config = measured
    report = build_serving_report(result, run_dir=run_dir)
    path = tmp_path / "report.json"

    write_serving_report(path, report, protected=[config])
    assert load_serving_report(path) == report
    with pytest.raises(ValueError, match="overwrite"):
        write_serving_report(path, report)
    with pytest.raises(ValueError, match="protected"):
        write_serving_report(config, report, protected=[config])
    tampered = json.loads(path.read_bytes())
    tampered["timing"]["restart_latency_ns"] += 1
    (tmp_path / "tampered.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="seal"):
        load_serving_report(tmp_path / "tampered.json")
    (tmp_path / "spaced.json").write_text(json.dumps(report, indent=1))
    with pytest.raises(ValueError, match="canonical"):
        load_serving_report(tmp_path / "spaced.json")
    with pytest.raises(ValueError, match="unsealed"):
        seal_serving_report(report)
