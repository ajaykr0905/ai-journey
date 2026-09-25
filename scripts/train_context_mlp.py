#!/usr/bin/env python3
"""Train a deterministic character context MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import load_corpus
from ai_journey.context_mlp import (
    build_context_dataset,
    dataset_fingerprint,
    model_fingerprint,
    parameter_count,
    train_context_mlp,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset = build_context_dataset(load_corpus(args.corpus))
    result = train_context_mlp(dataset, steps=args.steps, seed=args.seed)
    payload = {
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "model_fingerprint": model_fingerprint(result.model),
        "parameters": parameter_count(result.model),
        "steps": args.steps,
        "initial_loss": result.losses[0],
        "final_loss": result.losses[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
