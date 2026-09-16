"""Day 10: deterministic XOR training with a small NumPy network."""

from __future__ import annotations

from typing import Any

import numpy as np

from .neural_net import binary_cross_entropy, sigmoid


def train_xor(
    epochs: int = 5_000,
    learning_rate: float = 1.0,
    seed: int = 7,
) -> dict[str, Any]:
    """Train a 2-4-1 sigmoid network on XOR with full-batch gradient descent."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    generator = np.random.default_rng(seed)
    w1 = generator.normal(0.0, 0.7, size=(2, 4))
    b1 = np.zeros((1, 4), dtype=np.float64)
    w2 = generator.normal(0.0, 0.7, size=(4, 1))
    b2 = np.zeros((1, 1), dtype=np.float64)
    losses: list[float] = []

    for epoch in range(epochs + 1):
        a1 = sigmoid(inputs @ w1 + b1)
        predictions = sigmoid(a1 @ w2 + b2)
        if epoch == 0 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            losses.append(binary_cross_entropy(predictions, targets))
        if epoch == epochs:
            break

        dz2 = predictions - targets
        dw2 = a1.T @ dz2 / inputs.shape[0]
        db2 = np.mean(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ w2.T) * a1 * (1.0 - a1)
        dw1 = inputs.T @ dz1 / inputs.shape[0]
        db1 = np.mean(dz1, axis=0, keepdims=True)
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    labels = (predictions >= 0.5).astype(int)
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "sampled_losses": losses,
        "predictions": predictions,
        "labels": labels,
        "targets": targets.astype(int),
        "parameters": {"w1": w1, "b1": b1, "w2": w2, "b2": b2},
    }
