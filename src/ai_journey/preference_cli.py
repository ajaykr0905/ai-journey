"""Run and exactly replay the bounded preference-optimization comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .preference_data import PreferenceDataError
from .preference_experiment import (
    preference_config_from_dict,
    run_preference_comparison,
)
from .preference_report import (
    build_preference_report,
    replay_preference_report,
    write_preference_report,
)

CONFIG_BYTES = 16 * 1024
REPORT_BYTES = 8 * 1024 * 1024


def read_json(path, *, limit):
    """Read bounded UTF-8 JSON without duplicate or nonfinite values."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PreferenceDataError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise PreferenceDataError(f"nonfinite JSON literal: {value}")

    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PreferenceDataError(f"JSON exceeds its {limit}-byte budget")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PreferenceDataError(f"invalid preference JSON: {exc}") from exc


def load_config(path):
    return preference_config_from_dict(read_json(path, limit=CONFIG_BYTES))


def _preserve_config(config, output):
    if output.resolve() == config.resolve() or (
        output.exists() and config.exists() and output.samefile(config)
    ):
        raise PreferenceDataError("preference output must preserve its input config")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _preserve_config(args.config, args.output)
        report = build_preference_report(
            run_preference_comparison(load_config(args.config))
        )
        write_preference_report(args.output, report)
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        f"Compared {summary['observations']} seed-fold observations across "
        f"{len(summary['arms'])} policies; sha256={report['sha256']}; "
        f"output={args.output}"
    )
    return 0


def verify_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exactly replay a bounded CPU preference comparison.",
        allow_abbrev=False,
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verified = replay_preference_report(read_json(args.report, limit=REPORT_BYTES))
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Exactly replayed {verified['verified_bases']} bases, "
        f"{verified['verified_folds']} folds, and "
        f"{verified['verified_trials']} trials; sha256={verified['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
