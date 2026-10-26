from hashlib import sha256
import json
import subprocess
import pytest
from pathlib import Path

from ai_journey.ddp_launch import launch_training


def prepare_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/day-19-demo-names.txt"
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=4, micro_batch_size=1, accumulation_steps=2, sequence_length=4
    )
    payload["source"]["sha256"] = sha256(source.read_bytes()).hexdigest()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    return source, spec


def test_real_two_process_training_agrees_on_replicated_state(tmp_path):
    source, spec = prepare_run(tmp_path)
    ranks = launch_training(
        source=source,
        spec=spec,
        checkpoint=tmp_path / "state.pt",
        output_dir=tmp_path / "ranks",
        steps=2,
    )
    assert len({rank["pid"] for rank in ranks}) == 2
    assert len({rank["model_sha256"] for rank in ranks}) == 1
    assert all(rank["completed_step"] == 2 for rank in ranks)
    assert all(
        [step["tokens"] for step in rank["metrics"]] == [16, 16] for rank in ranks
    )


@pytest.mark.parametrize(
    "mode,expected", [("exit", RuntimeError), ("stall", TimeoutError)]
)
def test_failed_or_stalled_worker_reaps_every_owned_process(
    tmp_path, monkeypatch, mode, expected
):
    source, spec = prepare_run(tmp_path)
    owned = []
    real_popen = subprocess.Popen

    def tracked(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked)
    with pytest.raises(expected):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=tmp_path / "state.pt",
            output_dir=tmp_path / "failure",
            steps=2,
            failure_rank=1,
            failure_mode=mode,
            timeout_seconds=1 if mode == "stall" else 10,
        )
    assert len(owned) == 2
    assert all(process.poll() is not None for process in owned)
    assert not (tmp_path / "state.pt").exists()


@pytest.mark.parametrize(
    "collision", ["source", "spec", "checkpoint", "symlink", "checkpoint-symlink"]
)
def test_worker_output_aliases_cannot_truncate_run_inputs(
    tmp_path, monkeypatch, collision
):
    source, spec = prepare_run(tmp_path)
    checkpoint = tmp_path / "state.pt"
    preserved = {}
    if collision == "source":
        copied = tmp_path / "rank-0.log"
        copied.write_bytes(source.read_bytes())
        source = copied
    elif collision == "spec":
        copied = tmp_path / "rank-1.json"
        copied.write_bytes(spec.read_bytes())
        spec = copied
    elif collision == "checkpoint":
        checkpoint = tmp_path / "rank-0.log"
        checkpoint.write_bytes(b"previous checkpoint")
    elif collision == "symlink":
        valuable = tmp_path / "valuable.log"
        valuable.write_bytes(b"keep caller log")
        (tmp_path / "rank-0.log").symlink_to(valuable)
        preserved[valuable] = valuable.read_bytes()
    else:
        valuable = tmp_path / "valuable.pt"
        valuable.write_bytes(b"keep previous checkpoint")
        checkpoint.symlink_to(valuable)
        preserved[valuable] = valuable.read_bytes()
    for path in (source, spec, checkpoint):
        if path.exists():
            preserved[path] = path.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject before launching workers")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    expected = (
        "checkpoint output" if collision == "checkpoint-symlink" else "worker outputs"
    )
    with pytest.raises(ValueError, match=expected):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path,
            steps=2,
        )
    assert all(path.read_bytes() == value for path, value in preserved.items())
