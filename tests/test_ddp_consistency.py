from pathlib import Path
import copy

import pytest
import torch
import torch.multiprocessing as mp

from ai_journey.ddp_checkpoint import save_checkpoint, load_checkpoint
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import GPT2TrainingConfig
from ai_journey.ddp_training import train_segment
from torch.nn.parallel import DistributedDataParallel


def _checkpoint_replica(rank, directory, field):
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(48)
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    config = GPT2TrainingConfig(
        optimizer_steps=2,
        micro_batch_size=1,
        accumulation_steps=1,
        sequence_length=4,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0,
        gradient_clip=1,
        seed=48,
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    path = directory / "state.pt"
    with process_group(Topology(rank, 2), rendezvous=(directory / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        previous = path.read_bytes()
        if rank == 1:
            if field == "lr":
                optimizer.param_groups[0][field] *= 2
            else:
                optimizer.state[next(model.parameters())][field].add_(0.5)
        with pytest.raises(RuntimeError, match="optimizer fingerprints diverged"):
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
                step=1,
            )
        assert path.read_bytes() == previous
        assert sorted(item.name for item in directory.iterdir()) == ["rdzv", "state.pt"]


@pytest.mark.parametrize("field", ["exp_avg", "exp_avg_sq", "lr"])
def test_diverged_finite_adamw_state_preserves_the_last_checkpoint(tmp_path, field):
    mp.spawn(_checkpoint_replica, args=(str(tmp_path), field), nprocs=2, join=True)


def _assert_same(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_same(left, right)
    else:
        assert actual == expected


def _restore_replica(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(48)
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    config = GPT2TrainingConfig(
        optimizer_steps=2,
        micro_batch_size=1,
        accumulation_steps=1,
        sequence_length=4,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0,
        gradient_clip=1,
        seed=48,
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    path = directory / "state.pt"
    with process_group(Topology(rank, 2), rendezvous=(directory / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        if rank == 1:
            next(iter(payload["optimizer"]["state"].values()))["exp_avg"].add_(0.5)
        local_path = directory / f"state-{rank}.pt"
        torch.save(payload, local_path)
        with torch.no_grad():
            model.wte.weight.add_(rank + 1)
        torch.rand(rank + 1)
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="optimizer fingerprints diverged"):
            load_checkpoint(
                local_path,
                model=model,
                optimizer=optimizer,
                config=config,
                source=b"public",
            )
        _assert_same(model.state_dict(), before_model)
        _assert_same(optimizer.state_dict(), before_optimizer)
        assert torch.equal(torch.get_rng_state(), before_rng)


def test_rank_specific_checkpoint_moments_roll_back_restoration_everywhere(tmp_path):
    mp.spawn(_restore_replica, args=(str(tmp_path),), nprocs=2, join=True)


def _diverged_update(rank, directory):
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(48)
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)).double()

    class DivergingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            torch.rand(3)
            if rank == 1:
                with torch.no_grad():
                    model.wte.weight.add_(0.5)
            return result

    optimizer = DivergingAdamW(model.parameters(), lr=0.123)
    config = GPT2TrainingConfig(
        optimizer_steps=2,
        micro_batch_size=1,
        accumulation_steps=1,
        sequence_length=4,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0,
        gradient_clip=1,
        seed=48,
    )
    topology = Topology(rank, 2)
    with process_group(topology, rendezvous=(directory / "rdzv").as_uri()):
        wrapped = DistributedDataParallel(model).eval()
        model.h[0].train()
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = None if index % 2 else torch.full_like(parameter, 0.25)
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_rng = torch.get_rng_state().clone()
        gradients = [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]
        modes = [module.training for module in wrapped.modules()]
        with pytest.raises(RuntimeError, match="model fingerprints diverged"):
            train_segment(
                wrapped,
                optimizer,
                bytes(range(8)) * 4,
                config,
                topology,
                start_step=0,
                steps=1,
            )
        _assert_same(model.state_dict(), before_model)
        _assert_same(optimizer.state_dict(), before_optimizer)
        assert torch.equal(torch.get_rng_state(), before_rng)
        _assert_same([p.grad for p in model.parameters()], gradients)
        assert [module.training for module in wrapped.modules()] == modes


def test_divergent_update_restores_full_caller_state_on_both_ranks(tmp_path):
    mp.spawn(_diverged_update, args=(str(tmp_path),), nprocs=2, join=True)
