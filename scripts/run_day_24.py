#!/usr/bin/env python3
"""Train or resume the deterministic Day 24 transformer baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import (
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    run_transformer_experiment,
    write_experiment_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
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
    parser.add_argument("--seed", type=int, default=24)
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
    )
    training_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )
    result = run_transformer_experiment(
        corpus,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    write_experiment_report(args.output, result)
    print(
        f"steps={result.completed_steps} parameters={result.parameter_count} "
        f"train_nll={result.final_train_nll:.6f} "
        f"validation_nll={result.final_validation_nll:.6f}"
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
