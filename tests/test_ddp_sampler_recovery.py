"""Fresh CPU/Gloo processes retain versioned byte schedules across recovery."""

from dataclasses import replace
import json
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ai_journey.checkpoint_journal import CheckpointJournal
from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_checkpoint import load_checkpoint, save_checkpoint
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_launch import launch_training
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_training import train_segment
from ai_journey.gpt2_training import build_gpt2_optimizer, load_training_spec
from ai_journey.training_models import build_training_model
from test_ddp_sampler_versions import V2, public_files


def dropout_worker(rank, directory, phase, steps, resume):
    """Exercise the public training/checkpoint seam, not the zero-dropout spec."""
    directory = Path(directory)
    spec = load_training_spec(directory / "spec.json")
    stochastic = replace(
        spec.model, embedding_dropout=0.2, attention_dropout=0.2, residual_dropout=0.2
    )
    assert stochastic.embedding_dropout > 0
    torch.set_num_threads(1)
    torch.manual_seed(spec.training.seed)
    model = build_training_model(stochastic).double()
    optimizer = build_gpt2_optimizer(model, spec.training)
    topology = Topology(rank, 2)
    source = (directory / "public.bin").read_bytes()
    checkpoint = directory / (
        "continuous.pt" if phase == "continuous" else "restart.pt"
    )
    with process_group(
        topology,
        rendezvous=(directory / phase / "rendezvous").as_uri(),
        timeout_seconds=10,
    ):
        wrapped = DistributedDataParallel(model)
        start = (
            load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                config=spec.training,
                source=source,
                sampler=V2,
            )
            if resume
            else 0
        )
        metrics = train_segment(
            wrapped,
            optimizer,
            source,
            spec.training,
            topology,
            start_step=start,
            steps=steps,
            sampler=V2,
        )
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=spec.training,
            source=source,
            step=start + steps,
            sampler=V2,
        )
        (directory / phase / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "pid": __import__("os").getpid(),
                    "metrics": metrics,
                    "started_step": start,
                    "completed_step": start + steps,
                }
            )
        )


def test_public_training_dropout_restarts_exactly_in_fresh_ranks_across_epoch(tmp_path):
    public_files(tmp_path)
    for phase, steps, resume in (
        ("continuous", 4, False),
        ("first", 2, False),
        ("resumed", 2, True),
    ):
        (tmp_path / phase).mkdir()
        mp.spawn(
            dropout_worker,
            args=(str(tmp_path), phase, steps, resume),
            nprocs=2,
            join=True,
        )
    expected = torch.load(tmp_path / "continuous.pt", weights_only=True)
    actual = torch.load(tmp_path / "restart.pt", weights_only=True)
    assert expected["model_config"]["embedding_dropout"] == 0.2
    assert exact_state(actual, expected)
    for rank in range(2):
        records = [
            json.loads((tmp_path / phase / f"rank-{rank}.json").read_bytes())
            for phase in ("continuous", "first", "resumed")
        ]
        assert records[1]["pid"] != records[2]["pid"]
        assert records[2]["started_step"] == 2
        assert records[1]["metrics"] + records[2]["metrics"] == records[0]["metrics"]


def test_v2_journal_skips_a_byte_intact_wrong_contract_and_advances_high_water(
    tmp_path,
):
    source, spec = public_files(tmp_path)
    common = dict(source=source, spec=spec, sampler=V2)
    baseline = launch_training(
        **common,
        checkpoint=tmp_path / "continuous.pt",
        output_dir=tmp_path / "continuous",
        steps=4,
    )
    journal_dir = tmp_path / "journal"
    first = launch_training(
        **common,
        checkpoint=tmp_path / "restarted.pt",
        output_dir=tmp_path / "first",
        steps=2,
        journal_directory=journal_dir,
        checkpoint_interval=1,
    )
    manifest = json.loads((journal_dir / "manifest.json").read_bytes())
    journal = CheckpointJournal(journal_dir, run_sha256=manifest["run_sha256"])
    payload = torch.load(tmp_path / "restarted.pt", weights_only=True)
    payload["step"] = 3
    payload["data_contract"]["seed"] += 1
    invalid = tmp_path / "wrong-contract.pt"
    torch.save(payload, invalid)
    journal.publish(invalid, step=3)
    resumed = launch_training(
        **common,
        checkpoint=tmp_path / "restarted.pt",
        output_dir=tmp_path / "resumed",
        steps=2,
        resume=True,
        journal_directory=journal_dir,
        checkpoint_interval=1,
    )
    assert all(
        rank["journal_recovery"]["restored_step"] == 2
        and rank["journal_recovery"]["recorded_step"] == 3
        for rank in resumed
    )
    assert all(
        first[rank]["metrics"] + resumed[rank]["metrics"] == baseline[rank]["metrics"]
        for rank in range(2)
    )
    assert exact_state(
        torch.load(tmp_path / "continuous.pt", weights_only=True),
        torch.load(tmp_path / "restarted.pt", weights_only=True),
    )
    assert [
        entry["step"]
        for entry in json.loads((journal_dir / "manifest.json").read_bytes())["entries"]
    ] == [1, 2, 3, 4]
