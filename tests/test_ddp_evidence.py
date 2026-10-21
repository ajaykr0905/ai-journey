import copy
import json
from pathlib import Path
import pytest
from ai_journey.ddp_evidence import (
    build_evidence,
    write_evidence,
    read_evidence,
    verify_evidence,
    digest,
)
from ai_journey.ddp_launch import launch_training
from ai_journey.gpt2_training import load_training_spec


def measured_report(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/day-19-demo-names.txt"
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=2, sequence_length=4, micro_batch_size=1, accumulation_steps=2
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(payload))
    ranks = launch_training(
        source=source,
        spec=spec_path,
        checkpoint=tmp_path / "state.pt",
        output_dir=tmp_path / "ranks",
        steps=2,
    )
    return build_evidence(source.read_bytes(), load_training_spec(spec_path), ranks)


def test_measured_two_process_report_round_trips_and_detects_tampering(tmp_path):
    report = measured_report(tmp_path)
    path = tmp_path / "evidence.json"
    write_evidence(path, report)
    assert read_evidence(path) == report
    tampered = copy.deepcopy(report)
    tampered["metrics"][0]["tokens"] += 1
    with pytest.raises(ValueError, match="digest"):
        verify_evidence(tampered)
    tampered["evidence_sha256"] = digest(
        {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="accounting"):
        verify_evidence(tampered)
