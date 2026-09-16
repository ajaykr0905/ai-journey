#!/usr/bin/env python3
"""Generate and validate the Day 12 GPT tensor-shape reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_shapes import (  # noqa: E402
    TransformerConfig,
    build_gpt_shape_flow,
    parameter_count_breakdown,
    render_reference_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1_024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TransformerConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
    )
    flow = build_gpt_shape_flow(config)
    markdown = render_reference_markdown(flow)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    counts = parameter_count_breakdown(config)
    print(
        f"Validated {len(flow.tensors)} tensors and {len(flow.edges)} edges; "
        f"parameters={counts['total']}; output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
