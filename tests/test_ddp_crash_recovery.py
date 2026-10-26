import copy
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from ai_journey.ddp_acceptance import (
    certify_crash_recovery,
    crash_recovery_digest,
    read_crash_recovery,
    verify_crash_recovery,
)
from ai_journey.ddp_cli import main
from ai_journey.ddp_launch import launch_training
from ai_journey.ddp_operations import write_report
from ai_journey.ddp_worker import _write_boundary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def crash_evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-crash-recovery")
    workdir = directory / "owned"
    report = certify_crash_recovery(
        source=ROOT / "data/day-19-demo-names.txt",
        spec=ROOT / "config/ddp-lab.json",
        directory=workdir,
    )
    path = directory / "report.json"
    write_report(path, report)
    return report, path, workdir


def test_real_crash_recovers_in_fresh_workers_with_measured_rpo_rto(crash_evidence):
    report, path, workdir = crash_evidence
    assert read_crash_recovery(path, directory=workdir) == report
    assert report["injection"]["completed_step"] == 3
    assert report["last_durable"]["step"] == 2
    assert report["metrics"]["lost_optimizer_updates"] == 1
    assert report["metrics"]["resumed_optimizer_updates"] == 2
    assert (
        report["metrics"]["recovery_ready_upper_ns"]
        >= report["metrics"]["recovery_ready_lower_ns"]
        > 0
    )
    assert report["metrics"]["restoration_wall_ns"] > 0
    assert report["state_sha256"]["continuous"] == report["state_sha256"]["recovered"]
    failed = {row["pid"] for row in report["supervision"]["workers"]}
    resumed = {row["pid"] for row in report["resumed_workers"]}
    assert not failed & resumed


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["metrics"].update(lost_optimizer_updates=0),
        lambda r: r["metrics"].update(lost_optimizer_updates=True),
        lambda r: r["metrics"].update(
            restoration_wall_ns=float(r["metrics"]["restoration_wall_ns"])
        ),
        lambda r: r.update(world_size=True),
        lambda r: r["injection"].update(completed_step=4),
        lambda r: r["supervision"]["workers"][1].update(returncode=1),
        lambda r: r["supervision"].update(reaped_ns=r["resume_started_ns"] + 1),
        lambda r: r["restoration"][0].update(ready_ns=r["resume_started_ns"] - 1),
        lambda r: r["restoration"][0].update(model_sha256="0" * 64),
        lambda r: r["resumed_workers"][0].update(
            pid=r["supervision"]["workers"][0]["pid"]
        ),
        lambda r: r["state_sha256"]["recovered"].update(optimizer="0" * 64),
        lambda r: r.update(scope="production failover proof"),
    ],
)
def test_resealed_inconsistent_crash_evidence_is_rejected(crash_evidence, mutation):
    report = copy.deepcopy(crash_evidence[0])
    mutation(report)
    report["report_sha256"] = crash_recovery_digest(report)
    with pytest.raises(ValueError):
        verify_crash_recovery(report)


@pytest.mark.parametrize(
    "artifact",
    [
        "last-durable.pt",
        "continuous.pt",
        "recovered.pt",
        "failed/rank-1.crash.json",
        "resumed/rank-0.recovery.json",
    ],
)
def test_artifact_verification_recomputes_checkpoints_and_boundary_witnesses(
    crash_evidence, tmp_path, artifact
):
    report, _, original = crash_evidence
    copied = tmp_path / "copy"
    shutil.copytree(original, copied)
    path = copied / artifact
    if path.suffix == ".pt":
        payload = torch.load(path, weights_only=True)
        payload["step"] += 1
        torch.save(payload, path)
    else:
        payload = json.loads(path.read_text())
        payload["pid"] += 1
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifacts"):
        verify_crash_recovery(report, directory=copied)


def test_cli_verifies_actual_crash_certificate_and_artifacts(crash_evidence, capsys):
    report, path, workdir = crash_evidence
    assert (
        main(
            ["verify-crash-recovery", "--report", str(path), "--workdir", str(workdir)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == report


def test_recovery_report_fifo_cannot_block_waiting_for_a_writer(tmp_path):
    path = tmp_path / "report.fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        read_crash_recovery(path)


def test_crash_certificate_validates_source_budget_before_creating_artifacts(tmp_path):
    source = tmp_path / "oversize.txt"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    directory = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="budget"):
        certify_crash_recovery(
            source=source,
            spec=ROOT / "config/ddp-lab.json",
            directory=directory,
        )
    assert not directory.exists()


@pytest.mark.parametrize("payload", [b'{"schema":1,"schema":1}', b'{"value":NaN}'])
def test_recovery_json_rejects_ambiguous_or_nonfinite_fields(tmp_path, payload):
    path = tmp_path / "report.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_crash_recovery(path)


def test_boundary_writer_preserves_old_witness_on_publication_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "witness.json"
    path.write_bytes(b"keep old witness")

    def fail(*args):
        raise OSError("controlled replacement failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="controlled"):
        _write_boundary(path, {"public": True})
    assert path.read_bytes() == b"keep old witness"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("observe", [True, 1, None])
def test_restoration_observation_requires_an_explicit_resume(
    tmp_path, monkeypatch, observe
):
    def unexpected(*args, **kwargs):
        raise AssertionError("must reject before spawning any worker")

    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError, match="recovery observations"):
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "workers",
            steps=1,
            observe_recovery=observe,
        )
