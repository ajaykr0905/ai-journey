#!/usr/bin/env python3
"""Run the executable work for AI Journey Days 0 through 11."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ai_journey.drills import run_drills  # noqa: E402
from ai_journey.environment import environment_report  # noqa: E402
from ai_journey.gradients import gradient_descent_x_squared, save_loss_plot  # noqa: E402
from ai_journey.neural_net import (  # noqa: E402
    binary_cross_entropy,
    default_parameters,
    finite_difference_gradients,
    forward,
    full_backward,
    gradient_error,
    output_bias_gradient,
    output_weight_gradient,
    torch_autograd_gradients,
)
from ai_journey.xor import train_xor  # noqa: E402
from deployment_readiness_check import check_readiness, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory for generated, non-source artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    readiness = check_readiness(
        ROOT, load_config(ROOT / "config" / "deployment_readiness.json")
    )
    print(f"Day 00 | deployment ready: {readiness['ready']}")
    if not readiness["ready"]:
        failures.append("Day 0 deployment readiness")

    print("Day 01 | LLM lifecycle notes and project goal documented")

    environment = environment_report()
    print(
        "Day 02 | "
        f"Python {environment['python_version']}; "
        f"torch={environment['torch_available']}; "
        f"cuda={environment['cuda_available']}"
    )

    descent = gradient_descent_x_squared()
    plot_path = save_loss_plot(
        descent, args.artifact_dir / "day-03-gradient-descent.png"
    )
    print(
        "Day 03 | "
        f"loss {descent[0]['loss']:.6f} -> {descent[-1]['loss']:.6f}; "
        f"plot={plot_path if plot_path else 'skipped (matplotlib unavailable)'}"
    )
    if float(descent[-1]["loss"]) >= float(descent[0]["loss"]):
        failures.append("Day 3 gradient descent")

    print("Day 04 | two-input sigmoid-neuron derivation documented")

    inputs = np.array([[0.2, 0.7], [0.9, 0.1]], dtype=np.float64)
    targets = np.array([[1.0], [0.0]], dtype=np.float64)
    parameters = default_parameters()
    cache = forward(inputs, parameters)
    gradients = full_backward(cache, targets, parameters)
    print(
        "Day 05 | two-layer backward pass complete; "
        f"gradient shapes={{{', '.join(f'{k}: {v.shape}' for k, v in gradients.items())}}}"
    )
    print(
        "Day 06 | forward pass output="
        + np.array2string(cache["a2"].reshape(-1), precision=6)
    )
    print(
        "Day 07 | db2="
        + np.array2string(output_bias_gradient(cache, targets), precision=8)
    )
    print(
        "Day 08 | dw2="
        + np.array2string(output_weight_gradient(cache, targets), precision=8)
    )

    numerical = finite_difference_gradients(inputs, targets, parameters)
    numerical_errors = gradient_error(gradients, numerical)
    max_numerical_error = max(numerical_errors.values())
    torch_gradients = torch_autograd_gradients(inputs, targets, parameters)
    torch_summary: str
    if torch_gradients is None:
        torch_summary = "skipped (PyTorch unavailable)"
    else:
        torch_error = max(gradient_error(gradients, torch_gradients).values())
        torch_summary = f"max error={torch_error:.3e}"
        if torch_error >= 1e-9:
            failures.append("Day 9 torch.autograd gradient verification")
    print(
        "Day 09 | "
        f"finite-difference max error={max_numerical_error:.3e}; "
        f"torch.autograd={torch_summary}"
    )
    if max_numerical_error >= 1e-6:
        failures.append("Day 9 finite-difference gradient verification")

    xor_result = train_xor()
    xor_correct = bool(np.array_equal(xor_result["labels"], xor_result["targets"]))
    print(
        "Day 10 | "
        f"XOR loss {xor_result['initial_loss']:.6f} -> "
        f"{xor_result['final_loss']:.6f}; labels="
        f"{xor_result['labels'].reshape(-1).tolist()}"
    )
    if xor_result["final_loss"] >= xor_result["initial_loss"] or not xor_correct:
        failures.append("Day 10 XOR training")

    print("Day 11 | drills=" + json.dumps(run_drills(), sort_keys=True))

    if failures:
        print("FAILED | " + "; ".join(failures), file=sys.stderr)
        return 1
    print("Result | Days 0-11 executable checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
