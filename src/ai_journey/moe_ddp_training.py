"""Globally weighted replicated expert updates with one reducer iteration/window."""

import copy
import math
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .ddp_runtime import (
    require_all_ranks,
    require_model_consensus,
    require_optimizer_consensus,
)
from .decoder_variant_training import require_finite_decoder_state
from .moe_ddp_config import MoEDDPConfig
from .moe_ddp_resources import require_moe_ddp_budget
from .moe_ddp_window import MoEOptimizerWindow, validate_moe_ddp_window
from .moe_training import build_moe_optimizer, validate_moe_optimizer
from .moe_model import MoEDecoderModel
from .training_models import build_training_model
from .ddp_checkpoint import optimizer_run_contract


class MoEDDPFatalError(RuntimeError):
    """Tensor rollback does not make an interrupted reducer safe to reuse."""


class MoEDDPRetryableError(RuntimeError):
    """All backwards completed and a common rejection rolled back live data."""


class _CommonStepRejection(RuntimeError):
    pass


def validate_moe_ddp_optimizer(session):
    actual = optimizer_run_contract(session.model, session.optimizer)
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        session._optimizer_contract, sort_keys=True, allow_nan=False
    ) or any(
        group["lr"] != session.config.learning_rate
        for group in session.optimizer.param_groups
    ):
        raise ValueError(
            "MoE DDP optimizer differs from its declared fixed-rate protocol"
        )
    return validate_moe_optimizer(session.model, session.optimizer)


def _step_agreement(success, *, phase):
    try:
        require_all_ranks(success, phase=phase)
    except RuntimeError as error:
        if str(error) == f"at least one rank failed {phase}":
            raise _CommonStepRejection(str(error)) from error
        raise


class MoEDDPTrainingSession:
    """A CPU/Gloo replica with explicit completed-backward retry boundaries."""

    def __init__(self, config: MoEDDPConfig, *, model=None):
        if type(config) is not MoEDDPConfig:
            raise ValueError("MoE DDP session requires MoEDDPConfig")
        require_moe_ddp_budget(config)
        if (
            not dist.is_initialized()
            or dist.get_backend() != "gloo"
            or dist.get_world_size() != config.world_size
        ):
            raise ValueError("MoE DDP session requires its declared CPU/Gloo world")
        fingerprints = [None] * config.world_size
        dist.all_gather_object(fingerprints, config.fingerprint())
        require_all_ranks(len(set(fingerprints)) == 1, phase="MoE DDP run agreement")
        self.config = config
        if model is None:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(
                    torch.Generator(device="cpu").manual_seed(config.seed).get_state()
                )
                model = build_training_model(
                    config.model, max_parameters=1_000_000
                ).double()
        self.model = model
        model_valid = type(model) is MoEDecoderModel and model.config == config.model
        if model_valid:
            model_valid = all(
                parameter.device.type == "cpu" and parameter.dtype == torch.float64
                for parameter in model.parameters()
            )
        require_all_ranks(model_valid, phase="MoE DDP model admission")
        self.optimizer = build_moe_optimizer(
            self.model, learning_rate=config.learning_rate, config=config.optimizer
        )
        self._optimizer_contract = optimizer_run_contract(self.model, self.optimizer)
        self.wrapped = DistributedDataParallel(
            MoEOptimizerWindow(self.model, config), find_unused_parameters=True
        )
        self.poisoned = False

    def step(self, window):
        if self.poisoned:
            raise MoEDDPFatalError("MoE DDP session is poisoned; use fresh processes")
        count, valid = 0, True
        try:
            count = validate_moe_ddp_window(window, self.config)
            step = validate_moe_ddp_optimizer(self)
            require_finite_decoder_state(self.model.state_dict())
            if step >= self.config.optimizer_steps:
                raise ValueError("MoE DDP optimizer schedule is complete")
        except (ValueError, TypeError, RuntimeError):
            valid = False
        require_all_ranks(valid, phase="MoE DDP window admission")
        total = torch.tensor(count, dtype=torch.long)
        dist.all_reduce(total)
        require_all_ranks(total.item() > 0, phase="MoE DDP eligible targets")
        require_model_consensus(self.model)
        require_optimizer_consensus(self.optimizer)
        parameters = tuple(self.model.parameters())
        saved = (
            copy.deepcopy(self.model.state_dict()),
            copy.deepcopy(self.optimizer.state_dict()),
            torch.get_rng_state().clone(),
            [None if p.grad is None else p.grad.clone() for p in parameters],
        )
        modes = [(module, module.training) for module in self.wrapped.modules()]
        invoked = False
        backward_complete = False
        try:
            self.wrapped.train()
            self.optimizer.zero_grad(set_to_none=True)
            output = None
            invoked = True
            try:
                output = self.wrapped(window)
            except (ValueError, RuntimeError, TypeError):
                pass
            _step_agreement(output is not None, phase="MoE DDP forward")
            attempted, accepted = (
                output["attempted_counts"].clone(),
                output["accepted_counts"].clone(),
            )
            dist.all_reduce(attempted)
            dist.all_reduce(accepted)
            n, world = int(total), self.config.world_size
            layers, experts = (
                len(self.config.model.moe_layers),
                self.config.model.expert_count,
            )
            frequency = attempted.double() / (2 * n)
            local_balance = (
                experts * (frequency * output["probability_sums"]).sum() / (layers * n)
            )
            local_z = output["z_sums"].sum() / (layers * n)
            local_ce = output["ce_sum"] / n
            objective = world * (
                local_ce
                + self.config.balance_weight * local_balance
                + self.config.z_weight * local_z
            )
            _step_agreement(bool(torch.isfinite(objective)), phase="MoE DDP objective")
            measured = torch.stack(
                [local_ce.detach(), local_balance.detach(), local_z.detach()]
            )
            dist.all_reduce(measured)
            objective.backward()
            # Successful local backward alone is insufficient: every reducer
            # must have completed before a live-wrapper retry is authorized.
            _step_agreement(True, phase="MoE DDP backward completion")
            backward_complete = True
            idle = set()
            for index, layer in enumerate(self.config.model.moe_layers):
                for expert in range(experts):
                    if accepted[index, expert] == 0:
                        idle.update(
                            id(parameter)
                            for parameter in self.model.h[layer]
                            .mlp.experts[expert]
                            .parameters()
                        )
            finite = True
            for parameter in parameters:
                if parameter.grad is None and id(parameter) in idle:
                    parameter.grad = torch.zeros_like(parameter)
                finite = (
                    finite
                    and parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                )
            _step_agreement(finite, phase="MoE DDP finite active gradients")
            norm, after = None, None
            try:
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        self.config.max_gradient_norm,
                        error_if_nonfinite=True,
                    )
                )
                after = math.sqrt(
                    math.fsum(
                        float(parameter.grad.square().sum()) for parameter in parameters
                    )
                )
            except (ValueError, RuntimeError, TypeError, OverflowError):
                pass
            _step_agreement(
                norm is not None
                and after is not None
                and math.isfinite(norm)
                and math.isfinite(after),
                phase="MoE DDP gradient clipping",
            )
            stepped = True
            try:
                self.optimizer.step()
                require_finite_decoder_state(self.model.state_dict())
                require_finite_decoder_state(self.optimizer.state_dict())
                if validate_moe_ddp_optimizer(self) != step + 1:
                    raise ValueError("MoE DDP optimizer progress differs")
            except (ValueError, RuntimeError, TypeError):
                stepped = False
            _step_agreement(stepped, phase="MoE DDP optimizer update")
            try:
                fingerprint = require_model_consensus(self.model)
                require_optimizer_consensus(self.optimizer)
            except RuntimeError as error:
                if str(error) in (
                    "replicated model fingerprints diverged",
                    "replicated optimizer fingerprints diverged",
                ):
                    raise _CommonStepRejection(str(error)) from error
                raise
            ce, balance, z_loss = measured.tolist()
            return {
                "step": step + 1,
                "target_tokens": n,
                "ce_loss": ce,
                "balance_loss": balance,
                "router_z_loss": z_loss,
                "objective": ce
                + self.config.balance_weight * balance
                + self.config.z_weight * z_loss,
                "gradient_norm_before_clip": norm,
                "gradient_norm_after_clip": after,
                "model_sha256": fingerprint,
                "attempted_counts": attempted.tolist(),
                "accepted_counts": accepted.tolist(),
                "routing": output["routing"],
                "capacity_scope": "rank_microbatch",
            }
        except Exception as error:
            self.poisoned = invoked
            try:
                self.model.load_state_dict(saved[0], strict=True)
                self.optimizer.load_state_dict(saved[1])
                torch.set_rng_state(saved[2])
                for parameter, gradient in zip(parameters, saved[3], strict=True):
                    parameter.grad = gradient
            except Exception as rollback_error:
                raise MoEDDPFatalError(
                    "MoE DDP local rollback failed; use fresh processes"
                ) from rollback_error
            if invoked:
                if backward_complete and isinstance(error, _CommonStepRejection):
                    self.poisoned = False
                    raise MoEDDPRetryableError(
                        "MoE DDP update rejected after completed backward; live data restored"
                    ) from error
                raise MoEDDPFatalError(
                    "MoE DDP update failed; use fresh processes"
                ) from error
            raise
        finally:
            for module, mode in modes:
                module.training = mode
