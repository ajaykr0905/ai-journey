#!/usr/bin/env python3
"""Certify the Day 33 primitive WaveNet rebuild against its reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import build_vocabulary, load_corpus
from ai_journey.wavenet import (
    WaveNetConfig,
    WaveNetError,
    build_wavenet_dataset_split,
    initialize_wavenet,
)
from ai_journey.wavenet_certification import (
    certify_rebuild,
    write_certification_report,
)
from ai_journey.wavenet_rebuild import (
    initialize_rebuilt_wavenet,
    load_reference_parameters,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--group-factors", type=int, nargs="+", default=(2, 2))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=33)
    parser.add_argument("--seed", type=int, default=33)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.examples <= 0:
            raise WaveNetError("examples must be positive")
        words = load_corpus(args.corpus)
        config = WaveNetConfig(
            vocab_size=build_vocabulary(words).size,
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            group_factors=tuple(args.group_factors),
        )
        datasets = build_wavenet_dataset_split(
            words,
            config=config,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed,
        )
        if args.examples > datasets.train.sample_count:
            raise WaveNetError("examples exceeds the available training dataset")
        reference = initialize_wavenet(config, seed=args.seed)
        rebuilt = initialize_rebuilt_wavenet(config, seed=args.seed)
        load_reference_parameters(rebuilt, reference)
        contexts = datasets.train.contexts[: args.examples]
        targets = datasets.train.targets[: args.examples]
        result = certify_rebuild(reference, rebuilt, contexts, targets)
        if not result.passed:
            raise WaveNetError(
                f"rebuild certification failed: {', '.join(result.failed_gates)}"
            )
        write_certification_report(args.output, result)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Day 33 certification error: {exc}", file=sys.stderr)
        return 2

    print(
        f"gates=13 parameters={result.footprint.parameter_elements} "
        f"storage_bytes={result.footprint.total_bytes}"
    )
    print(f"model_fingerprint={result.model_fingerprint}")
    print(f"batch_fingerprint={result.batch_fingerprint}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
