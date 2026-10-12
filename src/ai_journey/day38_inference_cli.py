"""Generate text from an audited Day 38 checkpoint without retraining."""

import argparse
import sys
from pathlib import Path

from .day38_serving import (
    generate_from_artifact,
    load_trained_artifact,
    write_serving_report,
)
from .gpt_bigram import GPTBigramError
from .transformer_lab import TransformerLabError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "checkpoint", "corpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3800)
    parser.add_argument("--cached", action="store_true")
    return parser


def protect_output(output: Path, inputs: tuple[Path, ...]) -> None:
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise TransformerLabError("output must not overwrite training artifacts")
    if output.exists() and not output.is_file():
        raise TransformerLabError("output must be a regular file")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protect_output(args.output, (args.report, args.checkpoint, args.corpus))
        artifact = load_trained_artifact(args.report, args.checkpoint, args.corpus)
        report = generate_from_artifact(
            artifact,
            args.prompt,
            new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            cached=args.cached,
        )
        write_serving_report(args.output, report)
    except (GPTBigramError, TransformerLabError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
