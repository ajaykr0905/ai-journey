"""Build sealed reports for bounded public-data SFT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_dataset import load_chatml_dataset
from .sft_finetune_experiment import run_finetune_experiment
from .sft_finetune_io import write_finetune_report
from .sft_finetune_report import build_finetune_report


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SFTFineTuneError(f"duplicate config field: {key}")
        value[key] = item
    return value


def load_finetune_config(path: Path) -> SFTFineTuneConfig:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise SFTFineTuneError("config must be a bounded regular non-symlink file")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTFineTuneError("config is not strict UTF-8 JSON") from exc
    return SFTFineTuneConfig.from_mapping(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="run and seal a fine-tuning experiment")
    build.add_argument("--dataset", required=True, type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        inputs = {args.dataset.resolve(), args.config.resolve()}
        if args.output.resolve() in inputs:
            raise SFTFineTuneError("output must preserve dataset and config inputs")
        dataset = load_chatml_dataset(args.dataset.resolve())
        config = load_finetune_config(args.config.resolve())
        report = build_finetune_report(run_finetune_experiment(dataset, config))
        write_finetune_report(args.output.resolve(), report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "report_sha256": report["sha256"],
                "steps": config.steps,
                "delta_mean_nll": report["result"]["comparison"]["delta_mean_nll"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
