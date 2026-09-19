#!/usr/bin/env python3
"""Run and validate the deterministic Day 15 scalar autodiff experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.scalar_autodiff import (  # noqa: E402
    experiment_metrics,
    render_autodiff_markdown,
    run_autodiff_experiment,
    validate_experiment,
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
    experiment = run_autodiff_experiment()
    try:
        validate_experiment(experiment)
    except ValueError as exc:
        print(f"Day 15 validation failed: {exc}", file=sys.stderr)
        return 1

    metrics = experiment_metrics(experiment)
    markdown = render_autodiff_markdown(experiment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": metrics,
            "gradient_checks": [asdict(check) for check in experiment.gradient_checks],
        }
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print(
        "Validated Day 15 scalar autodiff: "
        f"nodes={metrics['node_count']}, edges={metrics['edge_count']}, "
        f"checks={metrics['gradient_check_count']}, "
        f"max_error={float(metrics['max_gradient_error']):.3e}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
