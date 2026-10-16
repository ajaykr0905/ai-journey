"""Compare tiny CPU decoder adaptation arms and replay their saved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .lora_data import LoRAExperimentError
from .lora_experiment import config_from_dict, run_comparison
from .lora_report import build_report, replay_report, write_report


CONFIG_BYTES = 16 * 1024
REPORT_BYTES = 256 * 1024


def read_json(path, *, limit):
    """Read bounded UTF-8 JSON, rejecting duplicate fields and nonfinite literals."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LoRAExperimentError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise LoRAExperimentError(f"nonfinite JSON literal: {value}")

    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise LoRAExperimentError(f"JSON exceeds its {limit}-byte budget")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise LoRAExperimentError(f"invalid comparison JSON: {exc}") from exc


def load_config(path):
    return config_from_dict(read_json(path, limit=CONFIG_BYTES))


def _preserve_config(config, output):
    if output.resolve() == config.resolve() or (
        output.exists() and config.exists() and output.samefile(config)
    ):
        raise LoRAExperimentError("comparison output must preserve its input config")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _preserve_config(args.config, args.output)
        report = build_report(run_comparison(load_config(args.config)))
        write_report(args.output, report)
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    seed_count = len(report["result"]["seeds"])
    print(
        f"Compared {seed_count} paired seeds / {seed_count * 4} CPU trials; "
        f"sha256={report['sha256']}; output={args.output}"
    )
    return 0


def verify_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exactly replay a bounded CPU LoRA comparison.", allow_abbrev=False
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verified = replay_report(read_json(args.report, limit=REPORT_BYTES))
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Exactly replayed {verified['verified_pretrains']} pretrained bases and "
        f"{verified['verified_trials']} CPU trials; sha256={verified['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
