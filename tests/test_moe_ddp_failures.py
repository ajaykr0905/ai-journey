"""Reducer lifecycle, not tensor rollback alone, decides if retry is supported."""

import copy
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.moe_ddp_training import (
    MoEDDPFatalError,
    MoEDDPRetryableError,
    MoEDDPTrainingSession,
)
from ai_journey.moe_training import build_moe_optimizer
from test_moe_ddp_training import run_config, serial_step, window


def failure_replica(rank, directory, kind):
    directory = Path(directory)
    torch.set_num_threads(1)
    config = run_config()
    with process_group(
        Topology(rank, 2), rendezvous=(directory / "rdzv").as_uri(), timeout_seconds=5
    ):
        session = MoEDDPTrainingSession(config)
        session.wrapped.eval()
        session.model.h[0].mlp.experts[0].train()
        for index, parameter in enumerate(session.model.parameters()):
            parameter.grad = None if index % 2 else torch.full_like(parameter, 0.25)
        torch.rand(rank + 1)
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
        reference = copy.deepcopy(session.model)
        reference_optimizer = build_moe_optimizer(
            reference, learning_rate=config.learning_rate, config=config.optimizer
        )
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step, original_forward = (
            session.optimizer.step,
            session.model.forward_training,
        )
        handle = None
        if rank == 1:
            if kind == "gradient":
                handle = session.model.wte.weight.register_hook(
                    lambda gradient: gradient * float("nan")
                )
            elif kind == "backward":

                def broken_backward(gradient):
                    raise RuntimeError("injected backward failure")

                handle = session.model.wte.weight.register_hook(broken_backward)
            elif kind == "forward":

                def broken_forward(*args, **kwargs):
                    raise ValueError("injected forward failure")

                session.model.forward_training = broken_forward
            elif kind == "clipping":

                def broken_clip(*args, **kwargs):
                    raise ValueError("injected clipping failure")

                torch.nn.utils.clip_grad_norm_ = broken_clip
            elif kind in ("step_raises", "moment_diverge", "model_diverge"):

                def broken_step(*args, **kwargs):
                    result = original_step(*args, **kwargs)
                    torch.rand(3)
                    if kind == "step_raises":
                        raise RuntimeError("injected post-step failure")
                    if kind == "model_diverge":
                        with torch.no_grad():
                            session.model.wte.weight.add_(0.25)
                    else:
                        session.optimizer.state[session.model.wte.weight][
                            "exp_avg"
                        ].add_(0.25)
                    return result

                session.optimizer.step = broken_step
        batches = window(rank)
        if kind == "preflight" and rank == 1:
            batches = batches[:-1]
        if kind == "empty_world":
            batches = tuple(
                (inputs, targets, torch.zeros_like(mask))
                for inputs, targets, mask in batches
            )
        expected_error = (
            MoEDDPFatalError
            if kind in ("forward", "backward")
            else RuntimeError
            if kind in ("preflight", "empty_world")
            else MoEDDPRetryableError
        )
        with pytest.raises(expected_error):
            session.step(batches)
        assert exact_state(session.model.state_dict(), before[0])
        assert exact_state(session.optimizer.state_dict(), before[1])
        assert torch.equal(torch.get_rng_state(), before[2])
        for parameter, gradient in zip(
            session.model.parameters(), before[3], strict=True
        ):
            assert (parameter.grad is None and gradient is None) or (
                parameter.grad is not None
                and gradient is not None
                and torch.equal(parameter.grad, gradient)
            )
        assert [module.training for module in session.wrapped.modules()] == before[4]
        assert all(
            not session.model.h[layer].mlp.router._forward_hooks
            for layer in config.model.moe_layers
        )
        if handle is not None:
            handle.remove()
        session.optimizer.step, session.model.forward_training = (
            original_step,
            original_forward,
        )
        torch.nn.utils.clip_grad_norm_ = original_clip
        if kind in ("forward", "backward"):
            assert session.poisoned
            with pytest.raises(MoEDDPFatalError, match="poisoned"):
                session.step(window(rank))
            return
        assert not session.poisoned
        for step in range(2):
            windows = [window(other, step=step) for other in range(2)]
            expected, _ = serial_step(reference, reference_optimizer, config, windows)
            actual = session.step(windows[rank])
            assert actual["objective"] == pytest.approx(
                expected["objective"], abs=1e-11, rel=1e-10
            )
            for name, parameter in session.model.named_parameters():
                torch.testing.assert_close(
                    parameter,
                    dict(reference.named_parameters())[name],
                    atol=1e-10,
                    rtol=1e-9,
                )
            assert actual["step"] == step + 1


@pytest.mark.parametrize(
    "kind",
    [
        "preflight",
        "empty_world",
        "gradient",
        "clipping",
        "step_raises",
        "moment_diverge",
        "model_diverge",
        "forward",
        "backward",
    ],
)
def test_actual_rank_failures_enforce_proven_retry_or_fresh_process_boundary(
    tmp_path, kind
):
    mp.spawn(failure_replica, args=(str(tmp_path), kind), nprocs=2, join=True)
