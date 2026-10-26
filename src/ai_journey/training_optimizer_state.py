"""Validate durable AdamW progress before publishing or accepting restart state."""

import math

import torch


def validate_adamw_state(
    model, optimizer, *, step: int, execution_controls=None
) -> None:
    """Reject malformed finite moments and controls that cannot resume correctly."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("training checkpoints require AdamW")
    controls = {
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    if execution_controls is not None:
        if (
            not isinstance(execution_controls, dict)
            or set(execution_controls) != set(controls)
            or execution_controls["capturable"] is not False
            or execution_controls["differentiable"] is not False
            or any(
                execution_controls[name] is not None
                and execution_controls[name] is not False
                for name in ("foreach", "fused")
            )
        ):
            raise ValueError("declared CPU execution controls are unsupported")
        controls = execution_controls
    parameters = tuple(model.parameters())
    parameter_ids = {id(parameter) for parameter in parameters}
    bound = [p for group in optimizer.param_groups for p in group["params"]]
    if len(bound) != len(parameters) or {id(p) for p in bound} != parameter_ids:
        raise ValueError("AdamW must bind every model parameter exactly once")
    if any(id(parameter) not in parameter_ids for parameter in optimizer.state):
        raise ValueError("AdamW contains state for an unknown parameter")
    for group in optimizer.param_groups:
        # These controls are not configurable in the deterministic CPU protocol.
        # In particular, capturable=True can restore finite state successfully
        # yet fail only when AdamW attempts its next CPU update.
        for name, expected in controls.items():
            if (
                name not in group
                or type(group[name]) is not type(expected)
                or group[name] is not expected
            ):
                raise ValueError(
                    f"AdamW CPU execution control {name} must retain {expected!r}"
                )
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
