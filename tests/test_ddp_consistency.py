from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from ai_journey.ddp_checkpoint import save_checkpoint
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import GPT2TrainingConfig


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
