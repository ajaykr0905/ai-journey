#!/usr/bin/env python3
"""Run the Day 27 scratch BatchNorm train/eval mode experiment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.batchnorm_experiment import (
    BatchNormCriteria,
    evaluate_batchnorm_experiment,
    render_batchnorm_diagnostics,
    run_batchnorm_experiment,
    write_batchnorm_report,
)
from ai_journey.transformer_lab import TokenCorpus, TrainingConfig, TransformerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
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
    parser.add_argument("--batch-norm-eps", type=float, default=1e-5)
    parser.add_argument("--batch-norm-momentum", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--min-training-loss-reduction", type=float, default=0.01)
    parser.add_argument("--min-mode-nll-gap", type=float, default=1e-4)
    parser.add_argument("--min-train-batch-coupling", type=float, default=1e-4)
    parser.add_argument("--max-eval-batch-coupling", type=float, default=1e-7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = TokenCorpus.from_path(
        args.corpus,
        validation_fraction=args.validation_fraction,
        block_size=args.block_size,
    )
    result = run_batchnorm_experiment(
        corpus,
        model_config=TransformerConfig(
            vocab_size=corpus.vocab_size,
            block_size=args.block_size,
            embedding_dim=args.embedding_dim,
            head_count=args.heads,
            layer_count=args.layers,
            dropout=args.dropout,
            initialization_std=args.initialization_std,
            initialization_gain=args.kaiming_gain,
            normalization_mode="scratch_batch_norm",
            batch_norm_eps=args.batch_norm_eps,
            batch_norm_momentum=args.batch_norm_momentum,
        ),
        training_config=TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
        ),
    )
    criteria = BatchNormCriteria(
        min_training_loss_reduction=args.min_training_loss_reduction,
        min_mode_nll_gap=args.min_mode_nll_gap,
        min_train_batch_coupling=args.min_train_batch_coupling,
        max_eval_batch_coupling=args.max_eval_batch_coupling,
    )
    evaluation = evaluate_batchnorm_experiment(result, criteria)
    write_batchnorm_report(args.output, result, criteria)
    render_batchnorm_diagnostics(args.plot, result)
    print(
        f"initial_eval_nll={result.initial_eval_nll:.6f} "
        f"final_train_nll={result.final_train_nll:.6f}"
    )
    print(
        f"correct_eval_nll={result.mode_trap.eval_nll:.6f} "
        f"mistaken_train_nll={result.mode_trap.train_mode_nll:.6f} "
        f"absolute_gap={evaluation.absolute_mode_nll_gap:.6f}"
    )
    print(
        f"train_batch_coupling={evaluation.train_batch_coupling:.6f} "
        f"eval_batch_coupling={evaluation.eval_batch_coupling:.6f} "
        f"passed={evaluation.passed}"
    )
    if evaluation.violations:
        print(f"violations={','.join(evaluation.violations)}")
    print(f"report={args.output}")
    print(f"plot={args.plot}")
    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
