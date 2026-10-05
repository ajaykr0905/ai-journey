#!/usr/bin/env python3
"""Run a predeclared deterministic transformer ablation protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.ablation_protocol import (
    evaluate_ablation,
    load_ablation_protocol,
    run_ablation,
    write_ablation_report,
)
from ai_journey.transformer_lab import CharacterCodec, TokenCorpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.corpus.read_text(encoding="utf-8")
    vocab_size = len(CharacterCodec.from_text(text).tokens)
    protocol = load_ablation_protocol(args.protocol, vocab_size=vocab_size)
    corpus = TokenCorpus.from_path(
        args.corpus,
        validation_fraction=args.validation_fraction,
        block_size=protocol.baseline_arm.model_config.block_size,
    )
    result = run_ablation(corpus, protocol)
    evaluation = evaluate_ablation(result)
    write_ablation_report(args.output, result)
    print(
        f"protocol={protocol.name} arms={len(protocol.arms)} "
        f"paired_seeds={len(protocol.trial_seeds)} outcome={evaluation.outcome}"
    )
    for contrast in result.contrasts:
        print(
            f"contrast={contrast.arm_label}-vs-{contrast.baseline_label} "
            f"mean_delta={contrast.mean_delta:.9f}"
        )
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
