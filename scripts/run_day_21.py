#!/usr/bin/env python3
"""Compare two character-bigram corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import BigramValidationError, load_corpus
from ai_journey.corpus_shift import (
    CorpusShiftError,
    analyze_corpus_shift,
    report_payload,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "data" / "day-19-demo-names.txt",
    )
    parser.add_argument(
        "--shifted",
        type=Path,
        default=ROOT / "data" / "day-21-indian-cities.txt",
    )
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze_corpus_shift(
            load_corpus(args.baseline),
            load_corpus(args.shifted),
            smoothing=args.smoothing,
        )
    except (BigramValidationError, CorpusShiftError, OSError, TypeError) as exc:
        print(f"corpus shift failed: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(report_payload(report), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
