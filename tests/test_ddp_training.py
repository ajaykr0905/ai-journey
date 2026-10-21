import copy
import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.ddp_training import accumulated_update
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def make_model():
    torch.manual_seed(48)
    return GPT2LanguageModel(
        GPT2Config(
            8, 4, 8, 2, 1, embedding_dropout=0, attention_dropout=0, residual_dropout=0
        )
    ).double()


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
        with pytest.raises(ValueError, match="micro-batch"):
            accumulated_update(
                DistributedDataParallel(model),
                torch.optim.SGD(model.parameters(), lr=0.1),
                [],
            )
