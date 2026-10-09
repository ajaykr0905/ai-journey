"""Command-line runner for the verified Day 35 causal-average experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .causal_average import CausalAverageError
from .causal_experiment import (
    CausalExperimentConfig,
    build_experiment_report,
    run_causal_experiment,
    write_experiment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify causal prefix averaging through four equivalent methods."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--time", type=int, default=8)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = CausalExperimentConfig(
            batch_size=args.batch_size,
            time=args.time,
            channels=args.channels,
            seed=args.seed,
            tolerance=args.tolerance,
        )
        result = run_causal_experiment(config)
        report = build_experiment_report(result)
        write_experiment_report(args.output, report)
    except (CausalAverageError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    maximum_forward = max(
        result.forward.matmul_error,
        result.forward.softmax_error,
        result.forward.cumsum_error,
    )
    maximum_gradient = max(
        result.gradient.matmul_error,
        result.gradient.softmax_error,
        result.gradient.cumsum_error,
    )
    print(
        f"max_forward_error={maximum_forward:.3e} "
        f"max_gradient_error={maximum_gradient:.3e} "
        f"streaming_error={result.streaming_error:.3e}"
    )
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
