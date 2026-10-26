"""Run a bounded CPU comparison on two explicitly supplied public-safe byte files."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_budget import MAX_WORK_UNITS, require_comparison_budget
from .decoder_variant_experiment import run_variant_comparison
from .decoder_variant_report_io import write_new_report
from .gpt2_config import GPT2Config

MAX_INPUT_BYTES = 1024 * 1024
REPORT_SCHEMA_VERSION = 2
REPORT_SCOPE = "matched-byte experiment; held-out NLL is not a generalization claim"


def _bounded_integer(name: str, *, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{minimum}, {maximum}]"
            )
        return number

    return parse


def _learning_rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("learning_rate must be numeric") from exc
    if not math.isfinite(rate) or not 0 < rate <= 0.1:
        raise argparse.ArgumentTypeError("learning_rate must be finite and in (0, 0.1]")
    return rate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, required=True, help="public-safe training bytes"
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        required=True,
        help="distinct public-safe held-out bytes",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON report path"
    )
    parser.add_argument(
        "--seed",
        type=_bounded_integer("seed", minimum=0, maximum=2**63 - 1),
        default=52,
    )
    parser.add_argument(
        "--steps", type=_bounded_integer("steps", minimum=1, maximum=64), default=1
    )
    parser.add_argument(
        "--batch-size",
        type=_bounded_integer("batch_size", minimum=1, maximum=32),
        default=2,
    )
    parser.add_argument("--learning-rate", type=_learning_rate, default=0.001)
    parser.add_argument(
        "--context-length",
        type=_bounded_integer("context_length", minimum=2, maximum=64),
        default=4,
    )
    parser.add_argument(
        "--embedding-dim",
        type=_bounded_integer("embedding_dim", minimum=4, maximum=64),
        default=8,
    )
    parser.add_argument(
        "--head-count",
        type=_bounded_integer("head_count", minimum=1, maximum=16),
        default=2,
    )
    parser.add_argument(
        "--layer-count",
        type=_bounded_integer("layer_count", minimum=1, maximum=2),
        default=1,
    )
    parser.add_argument(
        "--evaluation-windows",
        type=_bounded_integer("evaluation_windows", minimum=1, maximum=256),
        default=4,
    )
    return parser


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        payload = stream.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    return payload


def _standard_variants(
    baseline: DecoderVariantConfig,
) -> dict[str, DecoderVariantConfig]:
    """Return the five fixed comparison arms in report order."""

    return {
        "baseline": baseline,
        "rmsnorm": replace(baseline, norm_kind="rmsnorm"),
        "swiglu": replace(baseline, mlp_kind="swiglu"),
        "rope": replace(baseline, position_kind="rope"),
        "combined": replace(
            baseline, norm_kind="rmsnorm", mlp_kind="swiglu", position_kind="rope"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Compare five decoder arms and create a stable, non-overwriting report."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in {"plan", "documents"}:
        from .decoder_variant_document_cli import main as documents_main

        return documents_main(
            ["run" if values[0] == "documents" else "plan", *values[1:]]
        )
    parser = _parser()
    args = parser.parse_args(values)
    work_units = (
        args.steps
        * args.batch_size
        * args.context_length
        * args.layer_count
        * args.embedding_dim
    )
    if work_units > MAX_WORK_UNITS:
        parser.error(f"requested experiment exceeds {MAX_WORK_UNITS} work units")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new path")
    try:
        baseline = DecoderVariantConfig(
            GPT2Config(
                vocab_size=256,
                context_length=args.context_length,
                embedding_dim=args.embedding_dim,
                head_count=args.head_count,
                layer_count=args.layer_count,
                embedding_dropout=0.0,
                attention_dropout=0.0,
                residual_dropout=0.0,
            )
        )
        variants = _standard_variants(baseline)
        require_comparison_budget(
            baseline.gpt2,
            steps=args.steps,
            batch_size=args.batch_size,
            evaluation_windows=args.evaluation_windows,
        )
        result = run_variant_comparison(
            train_bytes=_read_bytes(args.train),
            heldout_bytes=_read_bytes(args.heldout),
            variants=variants,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            evaluation_windows=args.evaluation_windows,
        )
        report = asdict(result)
        report.update(
            schema_version=REPORT_SCHEMA_VERSION,
            device="cpu",
            scope=REPORT_SCOPE,
            baseline_config=asdict(baseline.gpt2),
            learning_rate=args.learning_rate,
            evaluation_windows=args.evaluation_windows,
            torch_version=str(torch.__version__),
        )
        write_new_report(args.output, report)
    except (OSError, TypeError, ValueError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(f"report={args.output} device=cpu variants={len(result.variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
