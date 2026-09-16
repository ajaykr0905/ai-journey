"""Day 3 gradient-descent exercise."""

from __future__ import annotations

from pathlib import Path


def gradient_descent_x_squared(
    start: float = 4.0,
    learning_rate: float = 0.1,
    steps: int = 30,
) -> list[dict[str, float | int]]:
    """Minimize ``f(x) = x²`` using its analytic gradient ``2x``.

    Returns the starting state plus one state after each update.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    x = float(start)
    history: list[dict[str, float | int]] = [
        {"step": 0, "x": x, "loss": x * x, "gradient": 2.0 * x}
    ]
    for step in range(1, steps + 1):
        x -= learning_rate * (2.0 * x)
        history.append(
            {"step": step, "x": x, "loss": x * x, "gradient": 2.0 * x}
        )
    return history


def save_loss_plot(
    history: list[dict[str, float | int]], output_path: str | Path
) -> Path | None:
    """Save the loss curve when matplotlib is installed; otherwise return None."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(item["step"]) for item in history]
    losses = [float(item["loss"]) for item in history]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(steps, losses, marker="o", markersize=3)
    axis.set(title="Gradient descent on f(x) = x²", xlabel="Step", ylabel="Loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
