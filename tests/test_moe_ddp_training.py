"""Actual two-rank gradients match an independently assembled serial objective."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_config import MoEConfig
from ai_journey.moe_ddp_config import MoEDDPConfig, moe_ddp_config_from_mapping
from ai_journey.moe_ddp_resources import estimate_moe_ddp_job
from ai_journey.moe_ddp_training import MoEDDPTrainingSession
from ai_journey.moe_ddp_window import validate_moe_ddp_window
from ai_journey.moe_training import build_moe_optimizer
from ai_journey.training_models import build_training_model


def run_config(**kwargs):
    model = MoEConfig(
        GPT2Config(
            16,
            3,
            4,
            2,
            1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        ),
        capacity_factor=0.5,
    )
    return MoEDDPConfig(
        model, sequence_length=3, optimizer_steps=3, max_gradient_norm=1e6, **kwargs
    )


def window(rank, *, empty_rank=False, step=0):
    inputs = torch.tensor([[1 + rank, 3 + step, 5 + rank]])
    targets = (inputs + 1) % 16
    mask = torch.tensor([[True, rank == 0, False]])
    if empty_rank and rank == 1:
        mask.zero_()
    return ((inputs, targets, mask), (inputs.flip(1), targets.flip(1), mask.clone()))


def serial_step(model, optimizer, config, windows):
    """Reference loops rank/microbatch capacity groups, not one pooled batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    captured = {layer: [] for layer in config.model.moe_layers}
    handles = []
    for layer in config.model.moe_layers:

        def capture(module, args, output, layer=layer):
            captured[layer].append(output)

        handles.append(model.h[layer].mlp.router.register_forward_hook(capture))
    ce_terms, rows, count = [], [], 0
    try:
        for batches in windows:
            for inputs, targets, mask in batches:
                output = model.forward_training(inputs, token_mask=mask)
                ce_terms.append(
                    F.cross_entropy(output.logits[mask], targets[mask], reduction="sum")
                    if mask.any()
                    else output.logits[0, 0, 0] * 0
                )
                rows.extend(output.routing)
                count += int(mask.sum())
    finally:
        for handle in handles:
            handle.remove()
    probabilities, z_terms, accepted = [], [], []
    balance = torch.zeros((), dtype=torch.float64)
    for layer in config.model.moe_layers:
        selected = [row for row in rows if row["layer_index"] == layer]
        attempts = torch.tensor(
            [
                sum(row["attempted_counts"][expert] for row in selected)
                for expert in range(config.model.expert_count)
            ],
            dtype=torch.float64,
        )
        probability = sum(route.probabilities.sum(0) for route in captured[layer])
        probabilities.append(probability)
        balance = balance + config.model.expert_count * (
            (attempts / (2 * count)) * probability / count
        ).sum() / len(config.model.moe_layers)
        z_terms.extend(
            torch.logsumexp(route.logits.double(), -1).square().sum()
            for route in captured[layer]
        )
        accepted.append(
            [
                sum(row["accepted_counts"][expert] for row in selected)
                for expert in range(config.model.expert_count)
            ]
        )
    ce = sum(ce_terms) / count
    z_loss = sum(z_terms) / (count * len(config.model.moe_layers))
    objective = ce + config.balance_weight * balance + config.z_weight * z_loss
    objective.backward()
    idle = set()
    for index, layer in enumerate(config.model.moe_layers):
        for expert, accepted_count in enumerate(accepted[index]):
            if not accepted_count:
                idle.update(
                    id(parameter)
                    for parameter in model.h[layer].mlp.experts[expert].parameters()
                )
    for parameter in model.parameters():
        if parameter.grad is None and id(parameter) in idle:
            parameter.grad = torch.zeros_like(parameter)
        assert parameter.grad is not None
    gradients = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters()
    }
    optimizer.step()
    return {
        "ce_loss": float(ce.detach()),
        "balance_loss": float(balance.detach()),
        "router_z_loss": float(z_loss.detach()),
        "objective": float(objective.detach()),
    }, gradients


def gradient_oracle(rank, directory, empty_rank, layers):
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(53)
    config = run_config()
    if layers == 2:
        config = replace(
            config,
            model=replace(
                config.model,
                gpt2=replace(config.model.gpt2, layer_count=2),
                moe_layers=(0, 1),
            ),
        )
    model = build_training_model(config.model).double()
    with torch.no_grad():
        for index in config.model.moe_layers:
            model.h[index].mlp.router.projection.weight.zero_()
    reference = copy.deepcopy(model)
    reference_optimizer = build_moe_optimizer(
        reference, learning_rate=config.learning_rate, config=config.optimizer
    )
    with process_group(Topology(rank, 2), rendezvous=(directory / "rdzv").as_uri()):
        session = MoEDDPTrainingSession(config, model=model)
        observed = []

        def capture_gradients(optimizer, args, kwargs):
            observed.append(
                {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                }
            )

        handle = session.optimizer.register_step_pre_hook(capture_gradients)
        for step in range(2):
            windows = [
                window(other, empty_rank=empty_rank, step=step) for other in range(2)
            ]
            expected, gradients = serial_step(
                reference, reference_optimizer, config, windows
            )
            actual = session.step(windows[rank])
            for key, value in expected.items():
                assert actual[key] == pytest.approx(value, abs=1e-11, rel=1e-10)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    observed[-1][name], gradients[name], atol=1e-10, rtol=1e-9
                )
                torch.testing.assert_close(
                    parameter,
                    dict(reference.named_parameters())[name],
                    atol=1e-10,
                    rtol=1e-9,
                )
            assert len(session.optimizer.state) == len(tuple(model.parameters()))
            assert all(
                state["step"].item() == step + 1
                for state in session.optimizer.state.values()
            )
            assert actual["target_tokens"] == sum(
                int(batch[2].sum()) for batches in windows for batch in batches
            )
            assert all(
                len(module._forward_hooks) == 0
                for module in (
                    model.h[index].mlp.router for index in config.model.moe_layers
                )
            )
            if step == 0:
                assert all(counts[2:] == [0, 0] for counts in actual["accepted_counts"])
        handle.remove()
        assert not session.poisoned


@pytest.mark.parametrize("empty_rank,layers", [(False, 1), (True, 1), (False, 2)])
def test_replicated_global_gradients_and_dense_adamw_match_serial_windows(
    tmp_path, empty_rank, layers
):
    mp.spawn(
        gradient_oracle, args=(str(tmp_path), empty_rank, layers), nprocs=2, join=True
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", True),
        ("world_size", 3),
        ("accumulation_steps", 9),
        ("sequence_length", 4),
        ("learning_rate", float("nan")),
        ("balance_weight", -1),
        ("capacity_scope", "global_batch"),
    ],
)
def test_invalid_declared_windows_reject_allocation_free(field, value):
    with pytest.raises(ValueError):
        replace(run_config(), **{field: value})


def test_configuration_roundtrip_and_whole_window_admission():
    config = run_config()
    assert moe_ddp_config_from_mapping(config.to_mapping()) == config
    first = estimate_moe_ddp_job(config)
    doubled = estimate_moe_ddp_job(replace(config, accumulation_steps=4))
    assert (
        doubled["retained_route_elements_per_rank"]
        == 2 * first["retained_route_elements_per_rank"]
    )
    assert (
        doubled["retained_attention_elements_per_rank"]
        == 2 * first["retained_attention_elements_per_rank"]
    )
    assert doubled["whole_job_work_units"] > first["whole_job_work_units"]
    too_large = MoEConfig(GPT2Config(16, 500000, 1, 1, 1))
    with pytest.raises(ValueError):
        MoEDDPConfig(too_large)


def test_expanded_invalid_window_rejects_before_token_work():
    config = run_config()
    huge = torch.ones(1, dtype=torch.long).expand(1, 1_000_000_000)
    huge_mask = torch.ones(1, dtype=torch.bool).expand_as(huge)
    with pytest.raises(ValueError, match="shape/dtype/device"):
        validate_moe_ddp_window(((huge, huge, huge_mask),) * 2, config)


@pytest.mark.parametrize("mutation", ["extra", "missing", "huge_optimizer"])
def test_parser_rejects_inexact_or_unbounded_contracts(mutation):
    value = run_config().to_mapping()
    if mutation == "extra":
        value["unexpected"] = True
    elif mutation == "missing":
        del value["capacity_scope"]
    else:
        value["optimizer"]["beta1"] = 10**1000
    with pytest.raises(ValueError):
        moe_ddp_config_from_mapping(value)


def test_job_work_and_checkpoint_records_reject_before_model_allocation():
    large_work = MoEConfig(GPT2Config(256, 128, 64, 4, 2), expert_count=8)
    with pytest.raises(ValueError, match="whole_job_work_units"):
        MoEDDPConfig(
            large_work,
            micro_batch_size=4,
            accumulation_steps=8,
            sequence_length=128,
            optimizer_steps=16,
        )
    many_records = MoEConfig(
        GPT2Config(256, 4, 4, 1, 64), expert_count=64, moe_layers=tuple(range(64))
    )
    with pytest.raises(ValueError, match="checkpoint_records"):
        MoEDDPConfig(many_records, accumulation_steps=1, optimizer_steps=1)
