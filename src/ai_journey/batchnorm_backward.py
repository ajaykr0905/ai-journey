"""Inspect the reverse pass of training-mode, last-dimension BatchNorm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BatchNormBackwardTrace:
    values: dict[str, np.ndarray]
    gradients: dict[str, np.ndarray]


def _array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def manual_batchnorm_backward(
    inputs: object, scale: object, bias: object, upstream: object, *, eps: float = 1e-5
) -> BatchNormBackwardTrace:
    """Apply explicit chain-rule equations without calling autograd.

    All leading dimensions form the sample axis. The normalization uses biased
    batch variance. Running-statistic updates and inference mode are excluded.
    ``upstream`` seeds an arbitrary vector-Jacobian product, not just a sum loss.
    """
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a positive finite real number")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite real number")
    x = _array(inputs, "inputs")
    gamma, beta, dy = (
        _array(v, n)
        for v, n in ((scale, "scale"), (bias, "bias"), (upstream, "upstream"))
    )
    if x.ndim < 2 or x.shape[-1] == 0 or x.size == 0:
        raise ValueError("inputs must be nonempty with at least two dimensions")
    if gamma.shape != (x.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("scale and bias must match the last feature dimension")
    if dy.shape != x.shape:
        raise ValueError("upstream must have the input shape")
    axes = tuple(range(x.ndim - 1))
    count = x.size // x.shape[-1]
    if count <= 1:
        raise ValueError("training requires more than one sample per feature")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = x.mean(axis=axes)
        centered = x - mean
        squared = centered**2
        variance = squared.mean(axis=axes)
        inv_std = (variance + eps) ** -0.5
        normalized = centered * inv_std
        output = normalized * gamma + beta
        dnormalized = dy * gamma
        dinv_std = (dnormalized * centered).sum(axis=axes)
        dvariance = dinv_std * (-0.5) * (variance + eps) ** -1.5
        dsquared = np.broadcast_to(dvariance / count, x.shape).copy()
        dcentered = dnormalized * inv_std + 2 * centered * dsquared
        dmean = -dcentered.sum(axis=axes)
        dx = dcentered + dmean / count
        values = {
            "inputs": x,
            "scale": gamma,
            "bias": beta,
            "mean": mean,
            "centered": centered,
            "squared": squared,
            "variance": variance,
            "inv_std": inv_std,
            "normalized": normalized,
            "output": output,
        }
        gradients = {
            "inputs": dx,
            "scale": (dy * normalized).sum(axis=axes),
            "bias": dy.sum(axis=axes),
            "mean": dmean,
            "centered": dcentered,
            "squared": dsquared,
            "variance": dvariance,
            "inv_std": dinv_std,
            "normalized": dnormalized,
            "output": dy,
        }
    if not all(np.isfinite(v).all() for v in (*values.values(), *gradients.values())):
        raise ValueError("BatchNorm intermediates and gradients must be finite")
    return BatchNormBackwardTrace(values, gradients)


def audit_batchnorm_backward(
    inputs: object,
    scale: object,
    bias: object,
    upstream: object,
    *,
    eps: float = 1e-5,
    epsilon: float = 1e-6,
    tolerance: float = 1e-7,
) -> dict:
    """Compare every intermediate gradient and finite-difference leaf gradient.

    Finite differences cost two forward passes per leaf scalar. This diagnostic
    is intended for small CPU float64 examples, not the training hot path.
    """
    for name, value in (("epsilon", epsilon), ("tolerance", tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    trace = manual_batchnorm_backward(inputs, scale, bias, upstream, eps=eps)
    t = {
        name: torch.tensor(
            trace.values[name], dtype=torch.float64, device="cpu", requires_grad=True
        )
        for name in ("inputs", "scale", "bias")
    }
    axes = tuple(range(t["inputs"].ndim - 1))
    t["mean"] = t["inputs"].mean(dim=axes)
    t["centered"] = t["inputs"] - t["mean"]
    t["squared"] = t["centered"].square()
    t["variance"] = t["squared"].mean(dim=axes)
    t["inv_std"] = torch.rsqrt(t["variance"] + eps)
    t["normalized"] = t["centered"] * t["inv_std"]
    t["output"] = t["normalized"] * t["scale"] + t["bias"]
    for tensor in t.values():
        tensor.retain_grad()
    t["output"].backward(
        torch.tensor(trace.gradients["output"], dtype=torch.float64, device="cpu")
    )
    forward_errors = {
        name: float(np.max(np.abs(trace.values[name] - tensor.detach().numpy())))
        for name, tensor in t.items()
    }
    gradient_errors = {
        name: float(np.max(np.abs(trace.gradients[name] - tensor.grad.numpy())))
        for name, tensor in t.items()
    }
    numeric_errors = {}
    leaves = [trace.values[name] for name in ("inputs", "scale", "bias")]
    for leaf_index, name in enumerate(("inputs", "scale", "bias")):
        numeric = np.empty_like(leaves[leaf_index])
        for index in np.ndindex(numeric.shape):
            plus, minus = [v.copy() for v in leaves], [v.copy() for v in leaves]
            plus[leaf_index][index] += epsilon
            minus[leaf_index][index] -= epsilon
            if (
                plus[leaf_index][index] == leaves[leaf_index][index]
                or minus[leaf_index][index] == leaves[leaf_index][index]
            ):
                raise ValueError("epsilon cannot perturb leaf at this magnitude")
            yplus = manual_batchnorm_backward(*plus, upstream, eps=eps).values["output"]
            yminus = manual_batchnorm_backward(*minus, upstream, eps=eps).values[
                "output"
            ]
            numeric[index] = float(
                ((yplus - yminus) * trace.gradients["output"]).sum()
            ) / (2 * epsilon)
        numeric_errors[name] = float(np.max(np.abs(numeric - trace.gradients[name])))
    errors = [
        *forward_errors.values(),
        *gradient_errors.values(),
        *numeric_errors.values(),
    ]
    return {
        "forward_errors": forward_errors,
        "gradient_errors": gradient_errors,
        "finite_difference_errors": numeric_errors,
        "tolerance": tolerance,
        "passed": all(math.isfinite(e) and e <= tolerance for e in errors),
    }
