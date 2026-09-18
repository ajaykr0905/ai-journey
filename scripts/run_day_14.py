#!/usr/bin/env python3
"""Run and validate the deterministic Day 14 MLP memory experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.mlp_memory import (  # noqa: E402
    experiment_metrics,
    render_memory_markdown,
    run_memory_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional metrics JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = run_memory_experiment()
    metrics = experiment_metrics(experiment)
    markdown = render_memory_markdown(experiment, metrics)

    exact_checks = (
        bool(metrics["key_value_expansion_exact"])
        and float(metrics["direct_max_error"]) <= 1e-12
        and float(metrics["basis_invariance_max_error"]) <= 1e-12
        and bool(metrics["collision_increases_error"])
    )
    if not exact_checks:
        print("Day 14 validation failed", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    report = {
        "shapes": {
            "queries": list(experiment.queries.shape),
            "keys": list(experiment.weights.keys.shape),
            "values": list(experiment.weights.values.shape),
            "contributions": list(experiment.trace.contributions.shape),
            "feature_directions": list(experiment.feature_directions.shape),
        },
        "metrics": metrics,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print(
        "Validated Day 14 MLP memory: "
        f"direct_error={float(metrics['direct_max_error']):.3e}, "
        f"basis_error={float(metrics['basis_invariance_max_error']):.3e}, "
        f"sparse_mse={float(metrics['sparse_reconstruction_mse']):.6f}, "
        f"collision_mse={float(metrics['collision_reconstruction_mse']):.6f}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
