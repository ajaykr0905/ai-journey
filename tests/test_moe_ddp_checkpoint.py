"""Fresh real ranks preserve stochastic expert/data progression exactly."""

import copy
from hashlib import sha256
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.moe_checkpoint import _digest
from ai_journey.moe_ddp_checkpoint import (
    load_moe_ddp_checkpoint,
    save_moe_ddp_checkpoint,
)
from ai_journey.moe_ddp_training import MoEDDPTrainingSession
from test_moe_ddp_training import run_config, window

SOURCE = b"Public authored rank/window recipe fixture, version one."
CONTRACT = {
    "schedule": "authored rank/window function v1",
    "mask_policy": "unequal valid counts",
}


def reseal(payload):
    payload["checkpoint_sha256"] = _digest(
        {name: value for name, value in payload.items() if name != "checkpoint_sha256"}
    )


def restart_replica(rank, directory, phase):
    directory = Path(directory)
    torch.set_num_threads(1)
    config = run_config()
    config = replace(
        config,
        model=replace(
            config.model,
            gpt2=replace(
                config.model.gpt2,
                embedding_dropout=0.1,
                attention_dropout=0.1,
                residual_dropout=0.1,
            ),
        ),
    )
    with process_group(
        Topology(rank, 2), rendezvous=(directory / f"rdzv-{phase}").as_uri()
    ):
        session = MoEDDPTrainingSession(config)
        torch.set_rng_state(
            torch.Generator(device="cpu")
            .manual_seed(config.seed + rank + 1000)
            .get_state()
        )
        start = (
            load_moe_ddp_checkpoint(
                directory / "restart.pt",
                session=session,
                source=SOURCE,
                data_contract=CONTRACT,
            )
            if phase == "resume"
            else 0
        )
        assert start == (1 if phase == "resume" else 0)
        metrics = []
        for step in range(start, 1 if phase == "first" else 3):
            metrics.append(session.step(window(rank, step=step)))
        save_moe_ddp_checkpoint(
            directory / ("restart.pt" if phase == "first" else f"{phase}.pt"),
            session=session,
            source=SOURCE,
            data_contract=CONTRACT,
        )
        torch.save(
            {
                "model": session.model.state_dict(),
                "optimizer": session.optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "metrics": metrics,
                "pid": os.getpid(),
            },
            directory / f"{phase}-{rank}.pt",
        )


def test_fresh_two_rank_dropout_restart_matches_uninterrupted_state_and_metrics(
    tmp_path,
):
    for phase in ("baseline", "first", "resume"):
        mp.spawn(restart_replica, args=(str(tmp_path), phase), nprocs=2, join=True)
    for rank in (0, 1):
        baseline = torch.load(tmp_path / f"baseline-{rank}.pt", weights_only=True)
        resumed = torch.load(tmp_path / f"resume-{rank}.pt", weights_only=True)
        assert all(
            exact_state(baseline[name], resumed[name])
            for name in ("model", "optimizer", "rng")
        )
        assert exact_state(baseline["metrics"][1:], resumed["metrics"])
    payload = torch.load(tmp_path / "restart.pt", weights_only=True)
    assert payload["world_size"] == 2 and payload["step"] == 1
    assert payload["data_cursor"] == {"next_update_index": 1}
    assert len(payload["rank_rngs"]) == 2
    assert not torch.equal(payload["rank_rngs"][0], payload["rank_rngs"][1])
    assert payload["inner"]["optimizer_state"]["param_groups"][0]["foreach"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "world",
        "step",
        "cursor",
        "source",
        "data",
        "config",
        "rng",
        "controls",
        "moment",
        "archive",
    ],
)
def test_checkpoint_contract_rejection_preserves_caller_state(tmp_path, corruption):
    config = run_config(world_size=1)
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        session = MoEDDPTrainingSession(config)
        session.step(window(0))
        path = tmp_path / "state.pt"
        save_moe_ddp_checkpoint(
            path, session=session, source=SOURCE, data_contract=CONTRACT
        )
        payload = torch.load(path, weights_only=True)
        if corruption == "world":
            payload["world_size"] = True
        elif corruption == "step":
            payload["step"] = 1.0
        elif corruption == "cursor":
            payload["data_cursor"]["next_update_index"] = 2
        elif corruption == "source":
            payload["source_sha256"] = "0" * 64
        elif corruption == "data":
            payload["data_contract"]["schedule"] = "different data order"
        elif corruption == "config":
            payload["config"]["balance_weight"] = 0.2
        elif corruption == "rng":
            payload["rank_rngs"][0] = torch.zeros(3, dtype=torch.uint8)
        elif corruption == "controls":
            payload["inner"]["optimizer_state"]["param_groups"][0]["weight_decay"] = 0.4
            payload["inner"]["optimizer_contract"][0]["controls"]["weight_decay"] = 0.4
            reseal(payload["inner"])
        elif corruption == "moment":
            next(iter(payload["inner"]["optimizer_state"]["state"].values()))[
                "exp_avg"
            ] = torch.zeros(1)
            reseal(payload["inner"])
        reseal(payload)
        torch.save(payload, path)
        if corruption == "archive":
            path.write_bytes(b"not a tensor archive")
        with torch.no_grad():
            session.model.wte.weight.add_(0.5)
        torch.rand(3)
        session.wrapped.eval()
        before = (
            copy.deepcopy(session.model.state_dict()),
            copy.deepcopy(session.optimizer.state_dict()),
            torch.get_rng_state().clone(),
            [
                None if parameter.grad is None else parameter.grad.clone()
                for parameter in session.model.parameters()
            ],
            [module.training for module in session.wrapped.modules()],
        )
        with pytest.raises(
            RuntimeError, match="checkpoint validation|checkpoint restoration"
        ):
            load_moe_ddp_checkpoint(
                path, session=session, source=SOURCE, data_contract=CONTRACT
            )
        assert exact_state(session.model.state_dict(), before[0])
        assert exact_state(session.optimizer.state_dict(), before[1])
        assert torch.equal(torch.get_rng_state(), before[2])
        for parameter, gradient in zip(
            session.model.parameters(), before[3], strict=True
        ):
            assert (parameter.grad is None and gradient is None) or torch.equal(
                parameter.grad, gradient
            )
        assert [module.training for module in session.wrapped.modules()] == before[4]
        assert not session.poisoned


def rank_specific_snapshot(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    with process_group(Topology(rank, 2), rendezvous=(directory / "rdzv").as_uri()):
        session = MoEDDPTrainingSession(run_config())
        session.step(window(rank))
        path = directory / "state.pt"
        save_moe_ddp_checkpoint(
            path, session=session, source=SOURCE, data_contract=CONTRACT
        )
        payload = torch.load(path, weights_only=True)
        if rank == 1:
            payload["rank_rngs"][1] = (
                torch.Generator(device="cpu").manual_seed(1234).get_state()
            )
            reseal(payload)
        local = directory / f"state-{rank}.pt"
        torch.save(payload, local)
        before = (
            copy.deepcopy(session.model.state_dict()),
            copy.deepcopy(session.optimizer.state_dict()),
            torch.get_rng_state().clone(),
        )
        with pytest.raises(RuntimeError, match="snapshot agreement"):
            load_moe_ddp_checkpoint(
                local, session=session, source=SOURCE, data_contract=CONTRACT
            )
        assert exact_state(session.model.state_dict(), before[0])
        assert exact_state(session.optimizer.state_dict(), before[1])
        assert torch.equal(torch.get_rng_state(), before[2])


def test_rank_specific_resealed_checkpoint_snapshots_reject_before_mutation(tmp_path):
    mp.spawn(rank_specific_snapshot, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "field,value", [("weight_decay", 0.5), ("betas", (0.2, 0.7)), ("lr", 0.03)]
)
def test_runtime_optimizer_must_keep_declared_checkpoint_protocol(
    tmp_path, field, value
):
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        session = MoEDDPTrainingSession(run_config(world_size=1))
        session.optimizer.param_groups[0][field] = value
        with pytest.raises(RuntimeError, match="window admission"):
            session.step(window(0))
        with pytest.raises(RuntimeError, match="checkpoint admission"):
            save_moe_ddp_checkpoint(
                tmp_path / "state.pt",
                session=session,
                source=SOURCE,
                data_contract=CONTRACT,
            )
        assert not (tmp_path / "state.pt").exists()


def test_payload_seal_and_archive_identity_are_explicit_distinct_contracts(tmp_path):
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        session = MoEDDPTrainingSession(run_config(world_size=1))
        session.step(window(0))
        path = tmp_path / "state.pt"
        identity = save_moe_ddp_checkpoint(
            path, session=session, source=SOURCE, data_contract=CONTRACT
        )
        payload = torch.load(path, weights_only=True)
        assert identity.payload_sha256 == payload["checkpoint_sha256"]
        assert identity.archive_sha256 == sha256(path.read_bytes()).hexdigest()
        assert identity.archive_sha256 != identity.payload_sha256
        before = copy.deepcopy(session.model.state_dict())
        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_moe_ddp_checkpoint(
                path,
                session=session,
                source=SOURCE,
                data_contract=CONTRACT,
                expected_archive_sha256=identity.payload_sha256,
            )
        assert exact_state(session.model.state_dict(), before)
        assert (
            load_moe_ddp_checkpoint(
                path,
                session=session,
                source=SOURCE,
                data_contract=CONTRACT,
                expected_archive_sha256=identity.archive_sha256,
            )
            == 1
        )
