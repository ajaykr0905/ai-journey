"""Command-line build and verification for analytical inference evidence."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

from .inference_io import read_inference_report, write_inference_report
from .inference_protocol import InferenceProtocol
from .inference_report import build_inference_report
from .inference_seal import seal_inference_report
from .inference_speculative import SpeculativePlan

MAX_PROTOCOL_BYTES = 64 * 1024


def load_inference_protocol(path) -> InferenceProtocol:
    target = Path(path)
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PROTOCOL_BYTES:
        raise ValueError("protocol must be a bounded regular file")
    try:
        mapping = json.loads(target.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol must be valid UTF-8 JSON") from error
    if type(mapping) is not dict:
        raise ValueError("protocol must be a JSON object")
    values = dict(mapping)
    if type(values.get("request_lengths")) is not list:
        raise ValueError("request_lengths must be a JSON array")
    values["request_lengths"] = tuple(values["request_lengths"])
    speculative = values.get("speculative")
    if type(speculative) is not dict:
        raise ValueError("speculative must be a JSON object")
    values["speculative"] = SpeculativePlan(**speculative)
    return InferenceProtocol(**values)


def build_command(protocol_path, output_path) -> str:
    report = seal_inference_report(
        build_inference_report(load_inference_protocol(protocol_path))
    )
    write_inference_report(output_path, report)
    return report["report_sha256"]


def verify_command(report_path) -> str:
    return read_inference_report(report_path)["report_sha256"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="ai-journey-inference")
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--protocol", required=True)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", required=True)
    return result


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "build":
        digest = build_command(arguments.protocol, arguments.output)
    else:
        digest = verify_command(arguments.report)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
