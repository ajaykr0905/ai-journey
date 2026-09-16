"""A tiny two-layer sigmoid network with manual gradients."""

from __future__ import annotations

import numpy as np

Array = np.ndarray
Parameters = dict[str, Array]


def sigmoid(values: Array) -> Array:
    """Numerically stable-enough sigmoid for these bounded learning examples."""

    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def default_parameters() -> Parameters:
    """Return deterministic parameters for a 2-input, 2-hidden, 1-output net."""

    return {
        "w1": np.array([[0.50, -0.40], [0.30, 0.80]], dtype=np.float64),
        "b1": np.array([[0.10, -0.20]], dtype=np.float64),
        "w2": np.array([[0.70], [-1.10]], dtype=np.float64),
        "b2": np.array([[0.05]], dtype=np.float64),
    }


def forward(inputs: Array, parameters: Parameters) -> dict[str, Array]:
    """Run the deterministic two-layer forward pass and return its cache."""

    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("inputs must have shape (batch, 2)")
    z1 = x @ parameters["w1"] + parameters["b1"]
    a1 = sigmoid(z1)
    z2 = a1 @ parameters["w2"] + parameters["b2"]
    a2 = sigmoid(z2)
    return {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def binary_cross_entropy(predictions: Array, targets: Array) -> float:
    """Mean binary cross-entropy with clipping only at the loss boundary."""

    y_hat = np.clip(np.asarray(predictions, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(targets, dtype=np.float64).reshape(y_hat.shape)
    return float(-np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))


def output_bias_gradient(cache: dict[str, Array], targets: Array) -> Array:
    """Day 7: compute only dL/db2 for sigmoid plus binary cross-entropy."""

    y = np.asarray(targets, dtype=np.float64).reshape(cache["a2"].shape)
    return np.mean(cache["a2"] - y, axis=0, keepdims=True)


def output_weight_gradient(cache: dict[str, Array], targets: Array) -> Array:
    """Day 8: compute dL/dw2 for every output-layer weight."""

    y = np.asarray(targets, dtype=np.float64).reshape(cache["a2"].shape)
    dz2 = cache["a2"] - y
    return cache["a1"].T @ dz2 / y.shape[0]


def full_backward(
    cache: dict[str, Array], targets: Array, parameters: Parameters
) -> Parameters:
    """Day 9: manually backpropagate mean BCE through both layers."""

    y = np.asarray(targets, dtype=np.float64).reshape(cache["a2"].shape)
    batch_size = y.shape[0]
    dz2 = cache["a2"] - y
    dw2 = cache["a1"].T @ dz2 / batch_size
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
    da1 = dz2 @ parameters["w2"].T
    dz1 = da1 * cache["a1"] * (1.0 - cache["a1"])
    dw1 = cache["x"].T @ dz1 / batch_size
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
    return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


def finite_difference_gradients(
    inputs: Array,
    targets: Array,
    parameters: Parameters,
    epsilon: float = 1e-6,
) -> Parameters:
    """Estimate every parameter gradient with centered finite differences."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    estimates: Parameters = {}
    for name, values in parameters.items():
        gradient = np.zeros_like(values)
        for index in np.ndindex(values.shape):
            plus = {key: value.copy() for key, value in parameters.items()}
            minus = {key: value.copy() for key, value in parameters.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            plus_loss = binary_cross_entropy(forward(inputs, plus)["a2"], targets)
            minus_loss = binary_cross_entropy(forward(inputs, minus)["a2"], targets)
            gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
        estimates[name] = gradient
    return estimates


def gradient_error(
    analytic: Parameters, reference: Parameters
) -> dict[str, float]:
    """Return the maximum absolute error for each parameter tensor."""

    return {
        name: float(np.max(np.abs(analytic[name] - reference[name])))
        for name in analytic
    }


def torch_autograd_gradients(
    inputs: Array, targets: Array, parameters: Parameters
) -> Parameters | None:
    """Return torch.autograd gradients when PyTorch is installed."""

    try:
        import torch
    except ImportError:
        return None

    tensors = {
        name: torch.tensor(value.tolist(), dtype=torch.float64, requires_grad=True)
        for name, value in parameters.items()
    }
    x = torch.tensor(np.asarray(inputs).tolist(), dtype=torch.float64)
    y = torch.tensor(np.asarray(targets).reshape(-1, 1).tolist(), dtype=torch.float64)
    a1 = torch.sigmoid(x @ tensors["w1"] + tensors["b1"])
    a2 = torch.sigmoid(a1 @ tensors["w2"] + tensors["b2"])
    loss = torch.nn.functional.binary_cross_entropy(a2, y)
    loss.backward()
    return {
        name: np.asarray(tensor.grad.detach().cpu().tolist(), dtype=np.float64)
        for name, tensor in tensors.items()
    }
