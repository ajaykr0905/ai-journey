#!/usr/bin/env python3
"""Run and validate the deterministic Day 16 manual backprop experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.manual_backprop import (
    ManualGraphError,
    experiment_metrics,
    render_manual_backprop_markdown,
    run_manual_backprop_experiment,
    validate_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional JSON trace path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = run_manual_backprop_experiment()
    try:
        validate_experiment(experiment)
    except ManualGraphError as exc:
        print(f"Day 16 validation failed: {exc}", file=sys.stderr)
        return 1

    metrics = experiment_metrics(experiment)
    markdown = render_manual_backprop_markdown(experiment)
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
            json.dumps(
                {
                    "metrics": metrics,
                    "trace": asdict(experiment.trace),
                    "comparisons": [asdict(item) for item in experiment.comparisons],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        "Validated Day 16 manual backprop: "
        f"nodes={metrics['node_count']}, "
        f"reverse_steps={metrics['reverse_step_count']}, "
        f"accumulations={metrics['accumulation_step_count']}, "
        f"max_error={float(metrics['max_comparison_error']):.3e}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
