"""Validate durable AdamW progress before publishing or accepting restart state."""

import math

import torch


def validate_adamw_state(model, optimizer, *, step: int) -> None:
    """Reject malformed finite moments and controls that cannot resume correctly."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("training checkpoints require AdamW")
    parameters = tuple(model.parameters())
    parameter_ids = {id(parameter) for parameter in parameters}
    bound = [p for group in optimizer.param_groups for p in group["params"]]
    if len(bound) != len(parameters) or {id(p) for p in bound} != parameter_ids:
        raise ValueError("AdamW must bind every model parameter exactly once")
    if any(id(parameter) not in parameter_ids for parameter in optimizer.state):
        raise ValueError("AdamW contains state for an unknown parameter")
    for group in optimizer.param_groups:
        for name in ("lr", "weight_decay", "eps"):
            value = group[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "eps" and value == 0)
            ):
                raise ValueError(f"AdamW {name} must be finite and valid")
        betas = group["betas"]
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
                for value in betas
            )
        ):
            raise ValueError("AdamW beta values are invalid")
        if type(group["amsgrad"]) is not bool:
            raise ValueError("AdamW amsgrad must be boolean")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if not state and step == 0:
                continue
            expected = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                expected.add("max_exp_avg_sq")
            if set(state) != expected:
                raise ValueError("AdamW moment fields do not match")
            counter = state["step"]
            if (
                not isinstance(counter, torch.Tensor)
                or counter.numel() != 1
                or not torch.isfinite(counter).all()
                or counter.item() != step
            ):
                raise ValueError("AdamW counter does not match checkpoint progress")
            for name in expected - {"step"}:
                moment = state[name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or moment.device != parameter.device
                    or not torch.isfinite(moment).all()
                    or (name != "exp_avg" and (moment < 0).any())
                ):
                    raise ValueError("AdamW moment tensor is invalid")
