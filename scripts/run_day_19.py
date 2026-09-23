#!/usr/bin/env python3
"""Run and validate the deterministic Day 19 character bigram experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import (
    BigramValidationError,
    experiment_metrics,
    experiment_payload,
    load_corpus,
    manifest_payload,
    render_bigram_markdown,
    run_bigram_experiment,
)

DEFAULT_CORPUS_PATH = ROOT / "data" / "day-19-demo-names.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="UTF-8 corpus with one lowercase ASCII word per line",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1909)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path; stdout is used when omitted",
    )
    parser.add_argument("--json-output", type=Path, help="Optional full JSON report")
    parser.add_argument(
        "--manifest-output", type=Path, help="Optional corpus manifest JSON path"
    )
    return parser.parse_args(argv)


def _source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        corpus = load_corpus(args.corpus)
        experiment = run_bigram_experiment(
            corpus,
            sample_count=args.samples,
            seed=args.seed,
            source=_source_label(args.corpus),
        )
    except (BigramValidationError, OSError, TypeError, ValueError) as exc:
        print(f"Day 19 validation failed: {exc}", file=sys.stderr)
        return 1

    markdown = render_bigram_markdown(experiment)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        destination = str(args.output)
    else:
        print(markdown, end="")
        destination = "stdout"

    if args.json_output:
        _write_json(args.json_output, experiment_payload(experiment))
    if args.manifest_output:
        _write_json(args.manifest_output, manifest_payload(experiment.manifest))

    metrics = experiment_metrics(experiment)
    print(
        "Validated Day 19 character bigram model: "
        f"words={metrics['word_count']}, "
        f"vocabulary={metrics['vocabulary_size']}, "
        f"transitions={metrics['observed_transitions']}, "
        f"oracle_error={float(metrics['oracle_max_error']):.3e}, "
        f"terminated={metrics['terminated_samples']}/{metrics['sample_count']}, "
        f"output={destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
