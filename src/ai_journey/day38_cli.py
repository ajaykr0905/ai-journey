"""Train and audit the scale-limited Day 38 Tiny Shakespeare transformer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .day38_training import (
    Day38DataConfig,
    Day38TrainingConfig,
    build_day38_report,
    run_day38_experiment,
    write_day38_report,
)
from .gpt_bigram import (
    GPTBigramError,
    TINY_SHAKESPEARE,
    fetch_verified_corpus,
    load_verified_corpus,
)
from .transformer_lab import TransformerLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--train-token-limit", type=int, default=65_536)
    parser.add_argument("--validation-token-limit", type=int, default=8_192)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--sample-prompt", default="\n")
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=3_800)
    return parser


def _protect_paths(corpus: Path, output: Path, checkpoint: Path) -> None:
    paths = (corpus, output, checkpoint)
    for index, path in enumerate(paths):
        if path.exists() and not path.is_file():
            raise TransformerLabError("corpus and artifact paths must be regular files")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise TransformerLabError("artifact parent paths must be directories")
        for other in paths[:index]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                raise TransformerLabError(
                    "corpus, output and checkpoint paths must be distinct"
                )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _protect_paths(args.corpus, args.output, args.checkpoint)
        data_config = Day38DataConfig(
            block_size=args.block_size,
            train_token_limit=args.train_token_limit,
            validation_token_limit=args.validation_token_limit,
            validation_fraction=args.validation_fraction,
        )
        config = Day38TrainingConfig(
            data=data_config,
            embedding_dim=args.embedding_dim,
            head_count=args.heads,
            layer_count=args.layers,
            dropout=args.dropout,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            sample_prompt=args.sample_prompt,
            sample_tokens=args.sample_tokens,
            sample_temperature=args.sample_temperature,
            sample_top_k=args.sample_top_k,
            sample_seed=args.sample_seed,
        )
        text = (
            fetch_verified_corpus(args.corpus, TINY_SHAKESPEARE)
            if args.download
            else load_verified_corpus(args.corpus, TINY_SHAKESPEARE)
        )
        started = time.perf_counter()
        experiment = run_day38_experiment(
            text,
            TINY_SHAKESPEARE,
            config,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
        report = build_day38_report(
            experiment, elapsed_seconds=time.perf_counter() - started
        )
        write_day38_report(args.output, report)
    except (GPTBigramError, TransformerLabError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"steps={experiment.completed_steps} parameters={experiment.parameter_count} "
        f"train_nll={experiment.final_train_nll:.6f} "
        f"validation_nll={experiment.final_validation_nll:.6f}"
    )
    print(f"sample={experiment.sample!r}")
    print(f"checkpoint={args.checkpoint}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
