"""Bounded CPU transformer training on train-only BPE with exact restart."""

import argparse
from pathlib import Path

from .bpe import DEFAULT_PATTERN, TokenizerError
from .bpe_corpus import build_bpe_corpus
from .export_verification import verify_export
from .serving_benchmark import _publish
from .tokenizer_cli import read_documents
from .tokenizer_io import save_tokenizer
from .transformer_lab import (
    TransformerConfig,
    TrainingConfig,
    run_transformer_experiment,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "checkpoint", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=288)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--regex", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = [
            getattr(args, name).resolve()
            for name in ("train", "validation", "checkpoint", "tokenizer", "output")
        ]
        if len(set(paths)) != len(paths):
            raise TokenizerError("input and output paths must be distinct")
        if args.source_snapshot is not None and args.manifest is None:
            raise TokenizerError("source verification requires an export manifest")
        if args.manifest is not None:
            if any(
                path.resolve() in paths[2:]
                for path in (args.manifest, args.source_snapshot)
                if path is not None
            ):
                raise TokenizerError("output paths must preserve provenance inputs")
            verify_export(
                args.manifest,
                args.train,
                args.validation,
                source_path=args.source_snapshot,
            )
        if (
            not 256 <= args.vocab_size <= 512
            or not 1 <= args.steps <= 100
            or not 2 <= args.block_size <= 32
        ):
            raise TokenizerError(
                "CPU controls exceed vocabulary, step, or context limits"
            )
        train, validation = read_documents(args.train), read_documents(args.validation)
        if sum(len(text.encode()) for text in (*train, *validation)) > 64 * 1024:
            raise TokenizerError("CPU training text exceeds 64 KiB")
        corpus = build_bpe_corpus(
            train,
            validation,
            vocab_size=args.vocab_size,
            pattern=DEFAULT_PATTERN if args.regex else None,
        )
        if max(len(corpus.train_tokens), len(corpus.validation_tokens)) > 4096:
            raise TokenizerError("CPU token partitions exceed 4096 tokens")
        architecture = TransformerConfig(
            vocab_size=corpus.vocab_size,
            block_size=args.block_size,
            embedding_dim=16,
            head_count=2,
            layer_count=1,
        )
        training = TrainingConfig(steps=args.steps, batch_size=4, seed=40)
        result = run_transformer_experiment(
            corpus,
            model_config=architecture,
            training_config=training,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
        save_tokenizer(args.tokenizer, corpus.codec.tokenizer)
        _publish(
            args.output,
            {
                "schema_version": 1,
                "experiment": result.to_dict(),
                "runtime_scope": "bounded local CPU; no production or distributed evidence",
            },
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Verified {result.completed_steps} BPE transformer steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
