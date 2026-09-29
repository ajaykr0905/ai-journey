#!/usr/bin/env python3
"""Compare transformer activation and gradient health across initialization scales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.activation_experiment import (
    render_activation_histograms,
    render_gradient_histograms,
    run_initialization_comparison,
    write_comparison_report,
)
from ai_journey.transformer_lab import TokenCorpus, TrainingConfig, TransformerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--plot-module")
    parser.add_argument("--gradient-plot", type=Path, required=True)
    parser.add_argument("--gradient-parameter")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--initialization-std", type=float, default=0.02)
    parser.add_argument("--stressed-initialization-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = TokenCorpus.from_path(
        args.corpus,
        validation_fraction=args.validation_fraction,
        block_size=args.block_size,
    )
    model_config = TransformerConfig(
        vocab_size=corpus.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        head_count=args.heads,
        layer_count=args.layers,
        dropout=args.dropout,
        initialization_std=args.initialization_std,
    )
    training_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )
    result = run_initialization_comparison(
        corpus,
        model_config=model_config,
        training_config=training_config,
        stressed_initialization_std=args.stressed_initialization_std,
    )
    write_comparison_report(args.output, result)
    render_activation_histograms(args.plot, result, module_name=args.plot_module)
    render_gradient_histograms(
        args.gradient_plot,
        result,
        parameter_name=args.gradient_parameter,
    )
    for variant in result.variants:
        print(
            f"{variant.name}: initial_loss={variant.initial_snapshot.loss:.6f} "
            f"final_loss={variant.final_snapshot.loss:.6f} "
            f"initial_health={variant.initial_health.passed} "
            f"final_health={variant.final_health.passed}"
        )
    print(f"report={args.output}")
    print(f"plot={args.plot}")
    print(f"gradient_plot={args.gradient_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
