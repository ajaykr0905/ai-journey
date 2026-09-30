#!/usr/bin/env python3
"""Compare fixed-normal and fan-in Kaiming transformer initialization."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.initialization_comparison import (
    ComparisonCriteria,
    evaluate_comparison,
    render_initialization_audit,
    render_loss_curves,
    run_kaiming_comparison,
    write_comparison_report,
)
from ai_journey.transformer_lab import TokenCorpus, TrainingConfig, TransformerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-plot", type=Path, required=True)
    parser.add_argument("--audit-plot", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
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
    parser.add_argument("--kaiming-gain", type=float, default=math.sqrt(2.0))
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--max-relative-std-error", type=float, default=0.25)
    parser.add_argument("--min-variant-loss-reduction", type=float, default=0.01)
    parser.add_argument("--min-kaiming-mean-loss-improvement", type=float, default=0.01)
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
        initialization_gain=args.kaiming_gain,
    )
    training_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )
    criteria = ComparisonCriteria(
        max_relative_std_error=args.max_relative_std_error,
        min_variant_loss_reduction=args.min_variant_loss_reduction,
        min_kaiming_mean_loss_improvement=(args.min_kaiming_mean_loss_improvement),
    )
    result = run_kaiming_comparison(
        corpus,
        model_config=model_config,
        training_config=training_config,
    )
    evaluation = evaluate_comparison(result, criteria)
    write_comparison_report(args.output, result, criteria)
    render_loss_curves(args.loss_plot, result)
    render_initialization_audit(args.audit_plot, result)
    for variant in result.variants:
        print(
            f"{variant.name}: mean_loss={variant.loss_curve.mean_loss:.6f} "
            f"loss_reduction={variant.loss_curve.relative_loss_reduction:.6f} "
            f"final_train_nll={variant.final_train_nll:.6f} "
            f"final_validation_nll={variant.final_validation_nll:.6f}"
        )
    print(
        f"kaiming_mean_loss_improvement="
        f"{evaluation.kaiming_mean_loss_improvement:.6f} "
        f"max_relative_std_error={evaluation.maximum_relative_std_error:.6f} "
        f"passed={evaluation.passed}"
    )
    print(
        f"kaiming_validation_improved="
        f"{result.finding.kaiming_final_validation_nll_improved} "
        f"train_validation_tradeoff={result.finding.train_validation_tradeoff}"
    )
    if evaluation.violations:
        print(f"violations={','.join(evaluation.violations)}")
    print(f"report={args.output}")
    print(f"loss_plot={args.loss_plot}")
    print(f"audit_plot={args.audit_plot}")
    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
