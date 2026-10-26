from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_launch import launch_training


def _run_spec(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/day-19-demo-names.txt"
    payload = json.loads((root / "config/ddp-lab.json").read_text())
    payload["training"].update(optimizer_steps=4)
    payload["source"]["sha256"] = sha256(source.read_bytes()).hexdigest()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    return source, spec


def test_later_rank_failure_resumes_the_periodic_checkpoint_in_fresh_processes(
    tmp_path,
):
    source, spec = _run_spec(tmp_path)
    continuous = tmp_path / "continuous.pt"
    recovered = tmp_path / "recovered.pt"
    launch_training(
        source=source,
        spec=spec,
        checkpoint=continuous,
        output_dir=tmp_path / "continuous",
        steps=4,
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=recovered,
            output_dir=tmp_path / "interrupted",
            steps=4,
            checkpoint_interval=2,
            failure_rank=1,
            failure_mode="exit",
            failure_after_step=3,
        )
    checkpoint = torch.load(recovered, weights_only=True)
    assert checkpoint["step"] == 2
    assert "after optimizer step 3" in (tmp_path / "interrupted/rank-1.log").read_text()
    ranks = launch_training(
        source=source,
        spec=spec,
        checkpoint=recovered,
        output_dir=tmp_path / "resumed",
        steps=2,
        checkpoint_interval=2,
        resume=True,
    )
    assert all(
        rank["started_step"] == 2 and rank["completed_step"] == 4 for rank in ranks
    )
    actual = torch.load(recovered, weights_only=True)
    expected = torch.load(continuous, weights_only=True)
    assert all(
        exact_state(actual[name], expected[name])
        for name in ("model", "optimizer", "rank_rngs", "step")
    )


def test_final_partial_interval_is_checkpointed(tmp_path):
    source, spec = _run_spec(tmp_path)
    checkpoint = tmp_path / "state.pt"
    ranks = launch_training(
        source=source,
        spec=spec,
        checkpoint=checkpoint,
        output_dir=tmp_path / "ranks",
        steps=3,
        world_size=1,
        checkpoint_interval=2,
    )
    assert ranks[0]["completed_step"] == 3
    assert torch.load(checkpoint, weights_only=True)["step"] == 3
    previous = checkpoint.read_bytes()
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "invalid-resume",
            steps=2,
            world_size=1,
            checkpoint_interval=1,
            resume=True,
        )
    assert checkpoint.read_bytes() == previous


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_checkpoint_interval_stops_before_launch(
    tmp_path, monkeypatch, interval
):
    source, spec = _run_spec(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject before process creation")

    monkeypatch.setattr("subprocess.Popen", unexpected)
    with pytest.raises(ValueError, match="checkpoint_interval"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "run",
            checkpoint_interval=interval,
        )
