"""Reproduce the pinned Hindi/English held-out tokenizer experiment."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .bpe import DEFAULT_PATTERN, TokenizerError
from .parallel_corpus import align_and_split
from .serving_benchmark import _publish
from .tokenizer_experiment import run_compression_experiment
from .tokenizer_sources import HINDI, ENGLISH, load_public_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hindi", required=True, type=Path)
    parser.add_argument("--english", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--vocab-sizes", type=int, nargs=3, default=[272, 288, 304])
    parser.add_argument("--regex", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in (args.hindi.resolve(), args.english.resolve()):
            raise TokenizerError("experiment output must preserve source snapshots")
        split = align_and_split(
            load_public_corpus(args.hindi, HINDI),
            load_public_corpus(args.english, ENGLISH),
            train_count=args.train_count,
            seed=args.seed,
        )
        report = run_compression_experiment(
            split,
            vocab_sizes=tuple(args.vocab_sizes),
            pattern=DEFAULT_PATTERN if args.regex else None,
        )
        report["sources"] = [asdict(HINDI), asdict(ENGLISH)]
        report["split_seed"] = args.seed
        _publish(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified 6 tokenizer trials; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
