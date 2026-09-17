#!/usr/bin/env python3
"""Run and validate the deterministic Day 13 NumPy attention experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.attention import (  # noqa: E402
    AttentionConfig,
    make_random_attention_problem,
    max_reference_error,
    render_attention_markdown,
    single_head_attention,
    validate_attention_trace,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--non-causal",
        action="store_true",
        help="Allow every query to attend to every key",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional metrics JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AttentionConfig(
        sequence_length=args.sequence_length,
        model_dim=args.model_dim,
        head_dim=args.head_dim,
        seed=args.seed,
        causal=not args.non_causal,
    )
    inputs, projections = make_random_attention_problem(config)
    trace = single_head_attention(inputs, projections, causal=config.causal)
    metrics = validate_attention_trace(trace)
    reference_error = max_reference_error(
        trace, inputs, projections, causal=config.causal
    )
    markdown = render_attention_markdown(config, trace, metrics, reference_error)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    report = {
        "config": {
            "sequence_length": config.sequence_length,
            "model_dim": config.model_dim,
            "head_dim": config.head_dim,
            "seed": config.seed,
            "causal": config.causal,
        },
        "shapes": {
            "queries": list(trace.queries.shape),
            "scores": list(trace.scores.shape),
            "weights": list(trace.weights.shape),
            "context": list(trace.context.shape),
        },
        "metrics": {**metrics, "max_reference_error": reference_error},
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"Validated Day 13 attention: T={config.sequence_length}, "
        f"C={config.model_dim}, D={config.head_dim}, seed={config.seed}, "
        f"causal={config.causal}, reference_error={reference_error:.3e}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
