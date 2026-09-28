#!/usr/bin/env python3
"""Run the Day 24 exact-size transformer overfit capacity probe."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import (
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    run_overfit_probe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args(argv)

    corpus = TokenCorpus.from_path(args.corpus, block_size=args.block_size)
    result = run_overfit_probe(
        corpus,
        model_config=TransformerConfig(
            vocab_size=corpus.vocab_size,
            block_size=args.block_size,
            embedding_dim=args.embedding_dim,
            head_count=args.heads,
            layer_count=args.layers,
        ),
        training_config=TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        example_count=args.examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        f"examples={result.example_count} initial_nll={result.initial_nll:.6f} "
        f"final_nll={result.final_nll:.6f}"
    )
    return 0 if result.final_nll < result.initial_nll else 1


if __name__ == "__main__":
    raise SystemExit(main())
