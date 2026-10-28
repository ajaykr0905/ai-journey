"""Fresh supervised CPU/Gloo ranks recover actual immutable checkpoint history."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import threading

import pytest
import torch

from ai_journey.checkpoint_journal import _seal
from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_launch import launch_training
from test_ddp_periodic_recovery import _run_spec


def damage_latest(directory, *, semantic):
    manifest = directory / "manifest.json"
    value = json.loads(manifest.read_bytes())
    entry = value["entries"][-1]
    path = directory / entry["filename"]
    if not semantic:
        path.write_bytes(b"corrupted committed checkpoint")
        return
    # This re-sealed, storage-intact fixture proves that byte integrity is not
    # sufficient: a true CPU-incompatible control must fail full restoration.
    payload = torch.load(path, weights_only=True)
    payload["optimizer"]["param_groups"][0]["capturable"] = True
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    renamed = directory / f"step-{entry['step']:07d}-{digest}.pt"
    path.replace(renamed)
    entry.update(
        filename=renamed.name, sha256=digest, size_bytes=renamed.stat().st_size
    )
    value["manifest_sha256"] = _seal(
        {key: part for key, part in value.items() if key != "manifest_sha256"}
    )
    manifest.write_text(json.dumps(value))


@pytest.mark.parametrize("semantic", [False, True])
def test_two_rank_corrupt_latest_fallback_matches_uninterrupted_checkpoint(
    tmp_path, semantic
):
    source, spec = _run_spec(tmp_path)
    continuous, recovered, journal = (
        tmp_path / "continuous.pt",
        tmp_path / "recovered.pt",
        tmp_path / "journal",
    )
    baseline = launch_training(
        source=source,
        spec=spec,
        checkpoint=continuous,
        output_dir=tmp_path / "continuous",
        steps=4,
    )
    launch_training(
        source=source,
        spec=spec,
        checkpoint=recovered,
        output_dir=tmp_path / "initial",
        steps=2,
        checkpoint_interval=1,
        journal_directory=journal,
    )
    first_digest = json.loads((journal / "manifest.json").read_bytes())["entries"][0][
        "sha256"
    ]
    damage_latest(journal, semantic=semantic)
    # The convenience latest file is not the enabled journal's recovery source.
    recovered.write_bytes(b"not a usable legacy checkpoint")
    damaged_entry = json.loads((journal / "manifest.json").read_bytes())["entries"][-1]
    ranks = launch_training(
        source=source,
        spec=spec,
        checkpoint=recovered,
        output_dir=tmp_path / "resumed",
        steps=3,
        checkpoint_interval=1,
        resume=True,
        journal_directory=journal,
    )
    assert len({rank["pid"] for rank in ranks}) == 2
    assert all(
        rank["started_step"] == 1 and rank["completed_step"] == 4 for rank in ranks
    )
    assert all(
        rank["journal_recovery"]
        == {"restored_step": 1, "recorded_step": 2, "checkpoint_sha256": first_digest}
        for rank in ranks
    )
    assert all(rank["metrics"] == baseline[0]["metrics"][1:] for rank in ranks)
    actual, expected = (
        torch.load(recovered, weights_only=True),
        torch.load(continuous, weights_only=True),
    )
    assert all(
        exact_state(actual[key], expected[key])
        for key in ("model", "optimizer", "rank_rngs", "step")
    )
    entries = json.loads((journal / "manifest.json").read_bytes())["entries"]
    assert [entry["step"] for entry in entries] == [1, 2, 3, 4]
    assert entries[1] == damaged_entry  # History is neither replaced nor pruned.


def test_resume_must_finish_beyond_recorded_high_water_before_any_update(tmp_path):
    source, spec = _run_spec(tmp_path)
    checkpoint, journal = tmp_path / "state.pt", tmp_path / "journal"
    launch_training(
        source=source,
        spec=spec,
        checkpoint=checkpoint,
        output_dir=tmp_path / "initial",
        world_size=1,
        steps=2,
        checkpoint_interval=1,
        journal_directory=journal,
    )
    damage_latest(journal, semantic=False)
    before = checkpoint.read_bytes(), (journal / "manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "rejected",
            world_size=1,
            steps=1,
            resume=True,
            journal_directory=journal,
        )
    assert before == (checkpoint.read_bytes(), (journal / "manifest.json").read_bytes())
    assert (
        "worker schedule validation" in (tmp_path / "rejected/rank-0.log").read_text()
    )
    assert not (tmp_path / "rejected/rank-0.json").exists()

    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "missing-resume",
            world_size=1,
            steps=1,
            journal_directory=journal,
        )
    assert before == (checkpoint.read_bytes(), (journal / "manifest.json").read_bytes())
    assert (
        "new checkpoint journal admission"
        in (tmp_path / "missing-resume/rank-0.log").read_text()
    )


@pytest.mark.parametrize("mismatch", ["source", "world"])
def test_journal_run_mismatch_rejects_without_changing_durable_progress(
    tmp_path, mismatch
):
    source, spec = _run_spec(tmp_path)
    checkpoint, journal = tmp_path / "state.pt", tmp_path / "journal"
    launch_training(
        source=source,
        spec=spec,
        checkpoint=checkpoint,
        output_dir=tmp_path / "initial",
        world_size=1,
        steps=1,
        journal_directory=journal,
    )
    world_size = 2 if mismatch == "world" else 1
    if mismatch == "source":
        copied = tmp_path / "different-source.txt"
        copied.write_bytes(source.read_bytes() + b"public authored difference\n")
        source = copied
        value = json.loads(spec.read_bytes())
        value["source"]["sha256"] = sha256(source.read_bytes()).hexdigest()
        spec.write_text(json.dumps(value))
    before = checkpoint.read_bytes(), (journal / "manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "rejected",
            world_size=world_size,
            steps=1,
            resume=True,
            journal_directory=journal,
        )
    assert before == (checkpoint.read_bytes(), (journal / "manifest.json").read_bytes())
    assert (
        "checkpoint journal candidates"
        in (tmp_path / "rejected/rank-0.log").read_text()
    )


def test_shared_journal_is_leased_even_with_distinct_latest_and_worker_paths(
    tmp_path, monkeypatch
):
    source, spec = _run_spec(tmp_path)
    journal = tmp_path / "journal"
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def stop_spawn(*args, stdout, **kwargs):
        attempts.append(args)
        entered.set()
        assert release.wait(5)
        raise RuntimeError("controlled spawn stop")

    monkeypatch.setattr("subprocess.Popen", stop_spawn)
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(
            launch_training,
            source=source,
            spec=spec,
            checkpoint=tmp_path / "one.pt",
            output_dir=tmp_path / "one",
            world_size=1,
            steps=1,
            journal_directory=journal,
        )
        try:
            assert entered.wait(5)
            with pytest.raises(ValueError, match="already owned"):
                launch_training(
                    source=source,
                    spec=spec,
                    checkpoint=tmp_path / "two.pt",
                    output_dir=tmp_path / "two",
                    world_size=1,
                    steps=1,
                    journal_directory=journal,
                )
            assert len(attempts) == 1 and not (tmp_path / "two/rank-0.log").exists()
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="controlled spawn"):
                owner.result(timeout=5)


@pytest.mark.parametrize(
    "layout", ["worker", "worker_child", "worker_parent", "input", "symlink", "file"]
)
def test_unsafe_journal_paths_reject_before_spawning(tmp_path, monkeypatch, layout):
    source, spec = _run_spec(tmp_path)
    workers = tmp_path / "workers"
    journal = tmp_path / "journal"
    if layout == "worker":
        journal = workers
    elif layout == "worker_child":
        journal = workers / "journal"
    elif layout == "worker_parent":
        workers = journal / "workers"
    elif layout == "input":
        journal = tmp_path
    elif layout == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        journal.symlink_to(target, target_is_directory=True)
    else:
        journal.write_bytes(b"preserve caller data")
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: pytest.fail("unsafe journal path must reject before spawning"),
    )
    with pytest.raises(ValueError, match="journal directory"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=tmp_path / "state.pt",
            output_dir=workers,
            world_size=1,
            steps=1,
            journal_directory=journal,
        )
    if layout == "file":
        assert journal.read_bytes() == b"preserve caller data"


def test_cli_passes_journal_and_rejects_report_inside_it(tmp_path, monkeypatch, capsys):
    from ai_journey import ddp_cli

    source, spec = _run_spec(tmp_path)
    captured = []
    monkeypatch.setattr(
        ddp_cli, "launch_training", lambda **kwargs: captured.append(kwargs) or []
    )
    monkeypatch.setattr(
        ddp_cli, "build_evidence", lambda *args: {"public_fixture": True}
    )
    monkeypatch.setattr(ddp_cli, "write_evidence", lambda *args: None)
    journal, report = tmp_path / "journal", tmp_path / "report.json"
    argv = [
        "train",
        "--source",
        str(source),
        "--spec",
        str(spec),
        "--checkpoint",
        str(tmp_path / "state.pt"),
        "--workdir",
        str(tmp_path / "workers"),
        "--journal-directory",
        str(journal),
        "--output",
        str(report),
    ]
    assert ddp_cli.main(argv) == 0
    assert captured[0]["journal_directory"] == journal
    assert json.loads(capsys.readouterr().out) == {"public_fixture": True}
    argv[-1] = str(journal / "manifest.json")
    with pytest.raises(SystemExit) as error:
        ddp_cli.main(argv)
    assert error.value.code == 2 and len(captured) == 1


def test_impossible_version_schedule_rejects_before_any_checkpoint_or_update(tmp_path):
    source, spec = _run_spec(tmp_path)
    value = json.loads(spec.read_bytes())
    value["training"]["optimizer_steps"] = 17
    spec.write_text(json.dumps(value))
    checkpoint, journal = tmp_path / "state.pt", tmp_path / "journal"
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "rejected",
            world_size=2,
            steps=17,
            checkpoint_interval=1,
            journal_directory=journal,
            # The existing public failure boundary would witness an update if
            # capacity admission did not run before the training loop.
            failure_rank=1,
            failure_mode="crash",
            failure_after_step=1,
        )
    assert not checkpoint.exists()
    assert json.loads((journal / "manifest.json").read_bytes())["entries"] == []
    assert not (tmp_path / "rejected/rank-1.crash.json").exists()
    assert all(
        "checkpoint journal version admission"
        in (tmp_path / "rejected" / f"rank-{rank}.log").read_text()
        for rank in (0, 1)
    )


def test_resumed_capacity_admission_preserves_bytes_and_accepts_a_fitting_schedule(
    tmp_path,
):
    source, spec = _run_spec(tmp_path)
    value = json.loads(spec.read_bytes())
    value["training"]["optimizer_steps"] = 18
    spec.write_text(json.dumps(value))
    checkpoint, journal = tmp_path / "state.pt", tmp_path / "journal"
    launch_training(
        source=source,
        spec=spec,
        checkpoint=checkpoint,
        output_dir=tmp_path / "initial",
        world_size=1,
        steps=15,
        checkpoint_interval=1,
        journal_directory=journal,
    )
    before = (
        checkpoint.read_bytes(),
        {path.name: path.read_bytes() for path in journal.iterdir()},
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        launch_training(
            source=source,
            spec=spec,
            checkpoint=checkpoint,
            output_dir=tmp_path / "too-many-versions",
            world_size=1,
            steps=2,
            checkpoint_interval=2,
            journal_directory=journal,
            resume=True,
            failure_rank=0,
            failure_mode="crash",
            failure_after_step=16,
        )
    assert before == (
        checkpoint.read_bytes(),
        {path.name: path.read_bytes() for path in journal.iterdir()},
    )
    assert not (tmp_path / "too-many-versions/rank-0.crash.json").exists()
    assert (
        "checkpoint journal version admission"
        in (tmp_path / "too-many-versions/rank-0.log").read_text()
    )
    ranks = launch_training(
        source=source,
        spec=spec,
        checkpoint=checkpoint,
        output_dir=tmp_path / "fitting-resume",
        world_size=1,
        steps=2,
        journal_directory=journal,
        resume=True,
    )
    assert ranks[0]["started_step"] == 15
    assert ranks[0]["completed_step"] == 17
    entries = json.loads((journal / "manifest.json").read_bytes())["entries"]
    assert [entry["step"] for entry in entries] == [*range(1, 16), 17]
    assert torch.load(checkpoint, weights_only=True)["step"] == 17
