import copy
import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_training import accumulated_update
from ai_journey.ddp_training import train_segment
from ai_journey.gpt2_training import GPT2TrainingConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def make_model():
    torch.manual_seed(48)
    return GPT2LanguageModel(
        GPT2Config(
            8, 4, 8, 2, 1, embedding_dropout=0, attention_dropout=0, residual_dropout=0
        )
    ).double()


def test_schedule_segments_reconstruct_epoch_boundaries(tmp_path):
    model = make_model()
    config = GPT2TrainingConfig(
        optimizer_steps=3,
        micro_batch_size=1,
        accumulation_steps=2,
        sequence_length=4,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=0,
        weight_decay=0,
        gradient_clip=1,
        seed=48,
    )
    source = bytes(range(8)) * 3
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        metrics = train_segment(
            DistributedDataParallel(model),
            torch.optim.SGD(model.parameters(), lr=0.01),
            source,
            config,
            Topology(0, 1),
            start_step=0,
            steps=3,
        )
    assert [metric["step"] for metric in metrics] == [1, 2, 3]
    assert [metric["tokens"] for metric in metrics] == [8, 8, 8]
    assert metrics[0]["model_sha256"] != metrics[-1]["model_sha256"]


def test_nonfinite_gradient_cannot_advance_any_optimizer(tmp_path):
    model = make_model()
    original = copy.deepcopy(model.state_dict())
    hook = model.wte.weight.register_hook(lambda gradient: gradient * float("nan"))
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        wrapped = DistributedDataParallel(model)
        inputs = torch.tensor([[0, 1]])
        with pytest.raises(RuntimeError, match="finite gradient"):
            accumulated_update(
                wrapped, torch.optim.SGD(model.parameters(), lr=0.1), [(inputs, inputs)]
            )
    hook.remove()
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_ddp_accumulation_matches_a_complete_batch(tmp_path):
    model = make_model()
    reference = copy.deepcopy(model)
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = (inputs + 1) % 8
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
    torch.nn.functional.cross_entropy(
        reference(inputs).reshape(-1, 8), targets.reshape(-1)
    ).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    ref_opt.step()
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        result = accumulated_update(
            DistributedDataParallel(model),
            opt,
            [(inputs[:1], targets[:1]), (inputs[1:], targets[1:])],
        )
    assert result["tokens"] == 8
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_empty_accumulation_fails_before_optimizer_work(tmp_path):
    model = make_model()
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        with pytest.raises(RuntimeError, match="micro-batch"):
            accumulated_update(
                DistributedDataParallel(model),
                torch.optim.SGD(model.parameters(), lr=0.1),
                [],
            )


def test_partial_optimizer_failure_restores_parameters_rng_and_modes(tmp_path):
    model = make_model().eval()
    original = copy.deepcopy(model.state_dict())

    class FailingSGD(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            torch.rand(3)
            raise RuntimeError("injected optimizer failure")

    optimizer = FailingSGD(model.parameters(), lr=0.1)
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        wrapped = DistributedDataParallel(model)
        rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="optimizer update"):
            accumulated_update(
                wrapped, optimizer, [(torch.tensor([[0, 1]]), torch.tensor([[1, 2]]))]
            )
        assert torch.equal(torch.get_rng_state(), rng)
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
