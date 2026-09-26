#!/usr/bin/env python3
"""Run a deterministic context-model learning-rate search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import load_corpus
from ai_journey.context_checkpoint import save_checkpoint
from ai_journey.model_selection import (
    TrainingConfig,
    learning_rate_grid,
    run_model_selection,
    verify_selected_checkpoint,
    write_experiment_report,
    write_learning_rate_plot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--minimum-rate", type=float, default=0.001)
    parser.add_argument("--maximum-rate", type=float, default=0.2)
    parser.add_argument("--rate-count", type=int, default=7)
    args = parser.parse_args(argv)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=5,
        minimum_delta=1e-5,
        max_gradient_norm=1.0,
    )
    experiment = run_model_selection(
        load_corpus(args.corpus),
        config=config,
        learning_rates=learning_rate_grid(
            args.minimum_rate, args.maximum_rate, count=args.rate_count
        ),
    )
    write_experiment_report(args.output, experiment)
    if args.checkpoint is not None:
        save_checkpoint(
            args.checkpoint,
            experiment.selected.model,
            step=experiment.selected.best_epoch + 1,
        )
        verify_selected_checkpoint(args.checkpoint, experiment)
    if args.plot is not None:
        write_learning_rate_plot(args.plot, experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
