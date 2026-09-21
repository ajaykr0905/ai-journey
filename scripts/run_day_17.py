#!/usr/bin/env python3
"""Run and validate the deterministic Day 17 scalar MLP experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.tiny_mlp import (
    MLPValidationError,
    TrainingConfig,
    experiment_metrics,
    render_mlp_markdown,
    run_mlp_experiment,
    validate_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-coefficient", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainingConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2_coefficient=args.l2_coefficient,
        gradient_clip_norm=args.gradient_clip_norm,
        record_every=args.record_every,
        seed=args.seed,
    )
    try:
        experiment = run_mlp_experiment(config)
        validate_experiment(experiment)
    except (MLPValidationError, TypeError, ValueError) as exc:
        print(f"Day 17 validation failed: {exc}", file=sys.stderr)
        return 1

    metrics = experiment_metrics(experiment)
    markdown = render_mlp_markdown(experiment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"metrics": metrics, "experiment": asdict(experiment)}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    print(
        "Validated Day 17 scalar MLP: "
        f"parameters={metrics['parameter_count']}, "
        f"loss={float(metrics['initial_loss']):.6f}→"
        f"{float(metrics['final_loss']):.6f}, "
        f"accuracy={100.0 * float(metrics['final_accuracy']):.1f}%, "
        f"max_gradient_error={float(metrics['max_gradient_probe_error']):.3e}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
