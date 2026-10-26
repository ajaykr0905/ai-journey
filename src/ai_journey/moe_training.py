"""Transactional CPU expert updates with explicit dense AdamW state semantics."""

import copy
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder_variant_optimizer import DecoderOptimizerConfig, build_decoder_optimizer
from .decoder_variant_training import require_finite_decoder_state
from .moe_model import MoEDecoderModel
from .moe_route_audit import validate_moe_routing
from .training_optimizer_state import validate_adamw_state


def build_moe_optimizer(
    model,
    *,
    learning_rate: float,
    weight_decay: float = 0.01,
    config: DecoderOptimizerConfig | None = None,
):
    """Use exactly the dense comparison's matrix-decay/vector-no-decay builder."""
    if not isinstance(model, MoEDecoderModel):
        raise TypeError("model must be MoEDecoderModel")
    if config is None:
        config = DecoderOptimizerConfig(weight_decay=weight_decay)
    elif weight_decay != 0.01:
        raise ValueError("supply either explicit optimizer config or weight_decay")
    return build_decoder_optimizer(model, learning_rate=learning_rate, config=config)


def validate_moe_optimizer(model, optimizer) -> int:
    """Validate global counters without relaxing the existing dense defaults."""
    if not isinstance(optimizer, torch.optim.AdamW) or not optimizer.param_groups:
        raise ValueError("MoE training requires AdamW")
    steps = []
    for state in optimizer.state.values():
        counter = state.get("step")
        if (
            not isinstance(counter, Tensor)
            or counter.numel() != 1
            or not bool(torch.isfinite(counter).all())
        ):
            raise ValueError("MoE optimizer counter is invalid")
        value = counter.item()
        if (
            type(value) not in (int, float)
            or not 0 <= value <= 2**53
            or int(value) != value
        ):
            raise ValueError("MoE optimizer counter is invalid")
        steps.append(int(value))
    if len(set(steps)) > 1:
        raise ValueError("MoE AdamW counters must share the global update count")
    step = steps[0] if steps else 0
    controls = {
        name: optimizer.param_groups[0].get(name)
        for name in ("capturable", "differentiable", "foreach", "fused")
    }
    validate_adamw_state(model, optimizer, step=step, execution_controls=controls)
    return step


def _positive_number(value, name, *, allow_zero=False):
    if (
        type(value) not in (int, float)
        or not 0 <= value <= 1e6
        or not math.isfinite(value)
        or (not allow_zero and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )


def train_moe_step(
    model,
    optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    token_mask: Tensor | None = None,
    balance_weight: float = 0.01,
    z_weight: float = 0.0,
    max_gradient_norm: float = 1.0,
) -> dict:
    """Apply CE + weighted routing losses with complete failure rollback.

    Only parameters of experts with zero accepted routes receive synthetic zero
    gradients. AdamW still updates/decays every parameter each step: this is
    sparse forward compute, not sparse optimizer state or sparse weight decay.
    Caller RNG advances on success and is restored exactly on failure.
    """
    if not isinstance(model, MoEDecoderModel):
        raise TypeError("model must be MoEDecoderModel")
    for value, name, zero in (
        (balance_weight, "balance_weight", True),
        (z_weight, "z_weight", True),
        (max_gradient_norm, "max_gradient_norm", False),
    ):
        _positive_number(value, name, allow_zero=zero)
    if (
        not isinstance(inputs, Tensor)
        or not isinstance(targets, Tensor)
        or inputs.dtype != torch.long
        or targets.dtype != torch.long
        or inputs.ndim != 2
        or inputs.shape != targets.shape
        or not inputs.numel()
        or inputs.device.type != "cpu"
        or targets.device != inputs.device
    ):
        raise ValueError(
            "training requires matching nonempty CPU long input/target matrices"
        )
    mask = (
        torch.ones_like(inputs, dtype=torch.bool) if token_mask is None else token_mask
    )
    if (
        not isinstance(mask, Tensor)
        or mask.dtype != torch.bool
        or mask.shape != inputs.shape
        or mask.device != inputs.device
        or not bool(mask.any())
    ):
        raise ValueError("training token_mask must select at least one matching token")
    if targets[mask].min() < 0 or targets[mask].max() >= model.config.gpt2.vocab_size:
        raise ValueError("eligible targets contain an out-of-range ID")
    step = validate_moe_optimizer(model, optimizer)
    require_finite_decoder_state(model.state_dict())
    parameters = tuple(model.parameters())
    saved = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(),
        [None if p.grad is None else p.grad.clone() for p in parameters],
    )
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_training(inputs, token_mask=mask)
        for stats in output.routing:
            validate_moe_routing(stats, expert_count=model.config.expert_count)
        ce = F.cross_entropy(output.logits[mask], targets[mask])
        objective = (
            ce + balance_weight * output.balance_loss + z_weight * output.router_z_loss
        )
        if not bool(torch.isfinite(objective)):
            raise ValueError("MoE training objective must be finite")
        objective.backward()
        idle = set(output.idle_expert_parameters)
        for name, parameter in model.named_parameters():
            if parameter.grad is None and name in idle:
                parameter.grad = torch.zeros_like(parameter)
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("active model gradients must be present and finite")
        before = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, max_gradient_norm, error_if_nonfinite=True
            )
        )
        after = math.sqrt(
            math.fsum(float(p.grad.double().square().sum()) for p in parameters)
        )
        optimizer.step()
        require_finite_decoder_state(model.state_dict())
        require_finite_decoder_state(optimizer.state_dict())
        if validate_moe_optimizer(model, optimizer) != step + 1:
            raise ValueError("MoE optimizer did not advance exactly one update")
        result = {
            "ce_loss": float(ce.detach()),
            "balance_loss": float(output.balance_loss.detach()),
            "router_z_loss": float(output.router_z_loss.detach()),
            "objective": float(objective.detach()),
            "gradient_norm_before_clip": before,
            "gradient_norm_after_clip": after,
            "clipped": before > max_gradient_norm,
            "target_tokens": int(mask.sum()),
            "routing": output.routing,
        }
        optimizer.zero_grad(set_to_none=True)
        return result
    except Exception:
        model.load_state_dict(saved[0], strict=True)
        optimizer.load_state_dict(saved[1])
        torch.set_rng_state(saved[2])
        for parameter, gradient in zip(parameters, saved[3], strict=True):
            parameter.grad = gradient
        raise
    finally:
        for module, training in modes:
            module.training = training
