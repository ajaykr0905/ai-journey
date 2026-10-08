"""Installed command for the verified Day 34 Tiny Shakespeare baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gpt_bigram import (
    BigramTrainingConfig,
    GPTBigramError,
    TINY_SHAKESPEARE,
    build_checkpoint_payload,
    build_experiment_report,
    fetch_verified_corpus,
    load_verified_corpus,
    run_bigram_experiment,
    save_checkpoint,
    write_experiment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a deterministic character bigram on pinned Tiny Shakespeare."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = BigramTrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            sample_tokens=args.sample_tokens,
            sample_temperature=args.sample_temperature,
        )
        text = (
            fetch_verified_corpus(args.corpus, TINY_SHAKESPEARE)
            if args.download
            else load_verified_corpus(args.corpus, TINY_SHAKESPEARE)
        )
        experiment, model, optimizer, batcher, vocabulary = run_bigram_experiment(
            text, TINY_SHAKESPEARE, config
        )
        report = build_experiment_report(experiment)
        checkpoint = build_checkpoint_payload(
            model,
            optimizer,
            batcher,
            step=experiment.completed_step,
            vocabulary=vocabulary,
            source=TINY_SHAKESPEARE,
            config=config,
        )
        save_checkpoint(args.checkpoint, checkpoint)
        write_experiment_report(args.output, report)
    except (GPTBigramError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"train_nll={experiment.final_train_nll:.6f} "
        f"validation_nll={experiment.final_validation_nll:.6f} "
        f"steps={experiment.completed_step}"
    )
    print(f"report={args.output}")
    print(f"checkpoint={args.checkpoint}")
    return 0
