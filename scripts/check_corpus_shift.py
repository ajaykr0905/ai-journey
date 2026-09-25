#!/usr/bin/env python3
"""Fail when corpus drift exceeds an explicit policy."""

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
    ShiftPolicy,
    analyze_corpus_shift,
    assess_corpus_shift,
    assessment_payload,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-js-divergence", type=float, required=True)
    parser.add_argument("--max-perplexity-ratio", type=float, required=True)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze_corpus_shift(
            load_corpus(args.baseline),
            load_corpus(args.candidate),
            smoothing=args.smoothing,
        )
        assessment = assess_corpus_shift(
            report,
            ShiftPolicy(
                max_js_divergence=args.max_js_divergence,
                max_perplexity_ratio=args.max_perplexity_ratio,
            ),
        )
    except (BigramValidationError, CorpusShiftError, OSError, TypeError) as exc:
        print(f"corpus shift check failed: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(assessment_payload(report, assessment), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if assessment.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
