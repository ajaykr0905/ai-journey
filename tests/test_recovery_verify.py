import copy
import json
from hashlib import sha256

import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_report import build_recovery_report
from ai_journey.recovery_verify import verify_recovery_report


def evidence(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"public recovery fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(valid_config().to_mapping()))
    run_dir = tmp_path / "run"
    result = run_recovery_experiment(
        source_path=source, config_path=config, run_dir=run_dir
    )
    return build_recovery_report(result), run_dir / "trace.json"


def reseal(report):
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_verifier_recomputes_report_and_trace_relationships(tmp_path):
    report, trace = evidence(tmp_path)
    assert verify_recovery_report(report, trace_path=trace) == report


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("attempts", "resumed_attempt_id", "same", "distinct resumed"),
        ("attempts", "resumed_exit_code", 1, "exit codes"),
        ("fault", "mechanism", "exception", "mechanism"),
        ("loss_assessment", "replayed_items", 2, "does not reconcile"),
        ("loss_assessment", "omitted_items", 1, "does not reconcile"),
        ("timing", "recovery_time_ms", 0, "durations"),
    ],
)
def test_verifier_rejects_resealed_but_incoherent_evidence(
    tmp_path, section, field, value, message
):
    report, _ = evidence(tmp_path)
    changed = copy.deepcopy(report)
    if section == "attempts" and field == "resumed_attempt_id" and value == "same":
        value = changed["attempts"]["failed_attempt_id"]
    changed[section][field] = value
    reseal(changed)
    with pytest.raises(ValueError, match=message):
        verify_recovery_report(changed)


def test_verifier_rejects_digest_tampering_and_trace_mismatch(tmp_path):
    report, trace = evidence(tmp_path)
    changed = copy.deepcopy(report)
    changed["loss_assessment"]["final_items"] = 19
    with pytest.raises(ValueError, match="digest"):
        verify_recovery_report(changed)
    report["trace_sha256"] = "f" * 64
    reseal(report)
    with pytest.raises(ValueError, match="trace identity"):
        verify_recovery_report(report, trace_path=trace)
