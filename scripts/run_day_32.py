#!/usr/bin/env python3
"""Run the deterministic Day 32 hierarchical character-model experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import build_vocabulary, load_corpus
from ai_journey.wavenet import (
    WaveNetConfig,
    WaveNetTrainingConfig,
    build_wavenet_dataset_split,
    run_wavenet_experiment,
    write_wavenet_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--group-factors", type=int, nargs="+", default=(2, 2, 2))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=32)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=320)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        words = load_corpus(args.corpus)
        model_config = WaveNetConfig(
            vocab_size=build_vocabulary(words).size,
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            group_factors=tuple(args.group_factors),
            dropout=args.dropout,
        )
        training_config = WaveNetTrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
        )
        datasets = build_wavenet_dataset_split(
            words,
            config=model_config,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed,
        )
        result = run_wavenet_experiment(
            datasets,
            model_config=model_config,
            training_config=training_config,
            sample_seed=args.sample_seed,
        )
        write_wavenet_report(args.output, result)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Day 32 experiment error: {exc}", file=sys.stderr)
        return 2

    print(
        f"steps={result.completed_steps} parameters={result.parameter_count} "
        f"train_nll={result.final_train.nll:.6f} "
        f"validation_nll={result.final_validation.nll:.6f}"
    )
    print(f"sample={result.sample.text!r} terminated={result.sample.terminated}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
