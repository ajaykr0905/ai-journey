"""Generate actual loopback HTTP load evidence or verify an existing report offline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile

from .http_load import run_http_load
from .http_load_profile import HTTPWorkloadProfile, http_profile_from_mapping
from .http_load_report import build_http_load_report, verify_http_load_report
from .http_stream import strict_json


def _read_json(path, *, maximum):
    if path.is_symlink() or not path.is_file():
        raise ValueError("HTTP evidence input must be a regular non-symlink file")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("HTTP evidence input exceeds its byte budget")
    result = strict_json(payload)
    if type(result) is not dict:
        raise ValueError("HTTP evidence input must contain a JSON object")
    return result


def _write_new(path, payload):
    rendered = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser(
        "profile", help="write an editable bounded workload profile"
    )
    profile.add_argument("--output", type=Path, required=True)
    measure = commands.add_parser(
        "measure", help="send actual requests to an existing loopback service"
    )
    measure.add_argument("--endpoint", required=True)
    measure.add_argument("--profile", type=Path)
    measure.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="check a sealed report and recompute derived metrics offline"
    )
    verify.add_argument("--report", type=Path, required=True)
    readiness = commands.add_parser(
        "readiness",
        help="exercise this installed command against an owned tiny CPU server",
    )
    readiness.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "readiness":
            from .http_readiness import run_installed_http_readiness

            sibling = Path(sys.executable).with_name("ai-journey-http-load")
            summary = run_installed_http_readiness(
                args.output,
                command=sibling if sibling.is_file() else "ai-journey-http-load",
            )
            print(
                f"readiness={summary['report']} completed={summary['measured_requests']} output_tokens={summary['measured_output_tokens']}"
            )
        elif args.command == "verify":
            report = verify_http_load_report(
                _read_json(args.report, maximum=4 * 1024 * 1024)
            )
            print(
                f"verified={args.report} attempts={report['measurement']['totals']['attempted_requests']}"
            )
        else:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; choose a new path")
            if args.command == "profile":
                _write_new(args.output, HTTPWorkloadProfile().to_mapping())
                print(f"profile={args.output}")
            else:
                profile = (
                    http_profile_from_mapping(
                        _read_json(args.profile, maximum=64 * 1024)
                    )
                    if args.profile
                    else HTTPWorkloadProfile()
                )
                report = build_http_load_report(run_http_load(args.endpoint, profile))
                verify_http_load_report(report)
                _write_new(args.output, report)
                totals = report["measurement"]["totals"]
                print(
                    f"report={args.output} completed={totals['outcomes']['completed']} rejected={totals['outcomes']['rejected']} attempts={totals['attempted_requests']}"
                )
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        OverflowError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
