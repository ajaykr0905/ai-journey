"""Build sealed reports for bounded public-data SFT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .sft_finetune_dataset import load_chatml_dataset
from .sft_finetune_experiment import run_finetune_experiment
from .sft_finetune_io import load_finetune_report, write_finetune_report
from .sft_finetune_replay import replay_finetune_report
from .sft_finetune_report import build_finetune_report
from .sft_finetune_verify import verify_finetune_evidence


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
    for name, help_text in (
        ("verify", "verify a report and its exact dataset"),
        ("replay", "rerun and compare a sealed experiment"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--report", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = load_chatml_dataset(args.dataset.resolve())
        if args.command == "build":
            inputs = {args.dataset.resolve(), args.config.resolve()}
            if args.output.resolve() in inputs:
                raise SFTFineTuneError("output must preserve dataset and config inputs")
            config = load_finetune_config(args.config.resolve())
            report = build_finetune_report(run_finetune_experiment(dataset, config))
            write_finetune_report(args.output.resolve(), report)
            summary = {
                "report_sha256": report["sha256"],
                "steps": config.steps,
                "delta_mean_nll": report["result"]["comparison"]["delta_mean_nll"],
            }
        else:
            report = load_finetune_report(args.report.resolve())
            summary = (
                verify_finetune_evidence(report, dataset)
                if args.command == "verify"
                else replay_finetune_report(report, dataset)
            )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            summary,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
