#!/usr/bin/env python3
"""Run the deterministic Day 21 corpus-shift and boundary-integrity lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import BigramValidationError, load_corpus
from ai_journey.corpus_shift import (
    CorpusShiftValidationError,
    experiment_metrics,
    experiment_payload,
    render_corpus_shift_markdown,
    run_corpus_shift_experiment,
)

DEFAULT_BASELINE_PATH = ROOT / "data" / "day-19-demo-names.txt"
DEFAULT_SHIFTED_PATH = ROOT / "data" / "day-21-indian-cities.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-corpus",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="UTF-8 baseline corpus with one lowercase ASCII record per line",
    )
    parser.add_argument(
        "--shifted-corpus",
        type=Path,
        default=DEFAULT_SHIFTED_PATH,
        help="UTF-8 shifted corpus with one lowercase ASCII record per line",
    )
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional full JSON report")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = load_corpus(args.baseline_corpus)
        shifted = load_corpus(args.shifted_corpus)
        experiment = run_corpus_shift_experiment(
            baseline, shifted, smoothing=args.smoothing
        )
    except (
        BigramValidationError,
        CorpusShiftValidationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Day 21 validation failed: {exc}", file=sys.stderr)
        return 1

    markdown = render_corpus_shift_markdown(experiment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    if args.json_output:
        _write_json(args.json_output, experiment_payload(experiment))

    metrics = experiment_metrics(experiment)
    print(
        "Validated Day 21 corpus shift: "
        f"baseline={metrics['baseline_records']}, "
        f"shifted={metrics['shifted_records']}, "
        f"vocabulary={metrics['vocabulary_size']}, "
        f"js_divergence={float(metrics['transition_js_divergence']):.9f}, "
        f"cross_record_events={metrics['shifted_cross_record_events']}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
