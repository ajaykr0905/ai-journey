"""Run and exactly replay the bounded assistant-only chat SFT comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sft_data import SFTDataError
from .sft_experiment import run_sft_comparison, sft_config_from_dict
from .sft_report import build_sft_report, replay_sft_report, write_sft_report

CONFIG_BYTES = 16 * 1024
REPORT_BYTES = 2 * 1024 * 1024


def read_json(path, *, limit):
    """Read bounded UTF-8 JSON without duplicate or nonfinite values."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SFTDataError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise SFTDataError(f"nonfinite JSON literal: {value}")

    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise SFTDataError(f"JSON exceeds its {limit}-byte budget")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SFTDataError(f"invalid chat SFT JSON: {exc}") from exc


def load_config(path):
    return sft_config_from_dict(read_json(path, limit=CONFIG_BYTES))


def _preserve_config(config, output):
    if output.resolve() == config.resolve() or (
        output.exists() and config.exists() and output.samefile(config)
    ):
        raise SFTDataError("chat SFT output must preserve its input config")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _preserve_config(args.config, args.output)
        report = build_sft_report(run_sft_comparison(load_config(args.config)))
        write_sft_report(args.output, report)
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bases = len(report["result"]["seeds"])
    print(
        f"Compared {bases} paired bases / {bases * 4} CPU chat SFT trials; "
        f"sha256={report['sha256']}; output={args.output}"
    )
    return 0


def verify_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exactly replay a bounded CPU chat SFT comparison.",
        allow_abbrev=False,
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verified = replay_sft_report(read_json(args.report, limit=REPORT_BYTES))
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Exactly replayed {verified['verified_bases']} bases and "
        f"{verified['verified_trials']} CPU chat SFT trials; "
        f"sha256={verified['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
