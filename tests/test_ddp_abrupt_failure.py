import json
from pathlib import Path
import subprocess

import pytest
import torch

from ai_journey.ddp_launch import WorkerFailure, WorkerTimeout, launch_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("after_step", [None, 3])
def test_owned_worker_abrupt_exit_is_observed_and_every_child_is_reaped(
    tmp_path, monkeypatch, after_step
):
    owned = []
    popen = subprocess.Popen

    def tracked(*args, **kwargs):
        process = popen(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked)
    checkpoint = tmp_path / "state.pt"
    with pytest.raises(WorkerFailure, match="worker failed") as failed:
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=checkpoint,
            output_dir=tmp_path / "failed",
            steps=4,
            checkpoint_interval=2,
            failure_rank=1,
            failure_mode="crash",
            failure_after_step=after_step,
            timeout_seconds=20,
        )
    assert len(owned) == 2 and all(process.poll() is not None for process in owned)
    supervision = failed.value.supervision
    marker = json.loads((tmp_path / "failed/rank-1.crash.json").read_text())
    assert marker["schema"] == "ai-journey-ddp-crash-boundary-v1"
    assert marker["completed_step"] == (after_step or 0)
    assert marker["rank"] == 1 and marker["world_size"] == 2
    assert marker["exit_code"] == 86
    assert supervision["schema"] == "ai-journey-ddp-supervision-v1"
    assert not supervision["timed_out"]
    assert (
        supervision["started_ns"]
        <= marker["observed_ns"]
        <= supervision["failure_observed_ns"]
        <= supervision["reaped_ns"]
    )
    assert supervision["workers"][1]["pid"] == marker["pid"] == owned[1].pid
    assert supervision["workers"][1]["returncode"] == 86
    assert all(type(row["returncode"]) is int for row in supervision["workers"])
    assert all(
        supervision["started_ns"] <= row["exit_observed_ns"] <= supervision["reaped_ns"]
        for row in supervision["workers"]
    )
    if after_step is None:
        assert not checkpoint.exists()
    else:
        assert torch.load(checkpoint, weights_only=True)["step"] == 2


def test_stalled_worker_retains_deadline_and_reap_observations(tmp_path):
    with pytest.raises(WorkerTimeout) as failed:
        launch_training(
            source=ROOT / "data/day-19-demo-names.txt",
            spec=ROOT / "config/ddp-lab.json",
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "failed",
            steps=1,
            failure_rank=1,
            failure_mode="stall",
            timeout_seconds=1,
        )
    assert failed.value.supervision["timed_out"]
    assert len(failed.value.supervision["workers"]) == 2
    assert all(
        type(row["returncode"]) is int for row in failed.value.supervision["workers"]
    )
