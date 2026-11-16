"""Export verified lab traces, generate dashboards and verify both.

``export`` verifies each given recovery-lab run directory with its own lab
verifier, converts it to OpenTelemetry spans and publishes
``traces.otlp.json`` (an OTLP/JSON ExportTraceServiceRequest) and a sealed
``report.json`` into a new directory, then verifies what it wrote.
``dashboard`` publishes the Grafana dashboard JSON with its sealed report the
same way. ``verify`` reads a report, dispatches on its kind and recomputes it:
a trace export from the run directories of exactly its labs, a dashboard from
its embedded config; ``--config`` binds either to a pinned config file. Every
command prints one JSON receipt labeled as local CPU lab telemetry.

Exit codes: 0 success; 1 source evidence or a stored artifact was rejected;
2 usage errors, unreadable or invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .telemetry_dashboard_report import (
    REPORT_FILE as DASHBOARD_REPORT_FILE,
    REPORT_KIND as DASHBOARD_REPORT_KIND,
    build_dashboard_artifact,
    build_dashboard_report,
    verify_dashboard_report,
    write_dashboard_artifact,
)
from .telemetry_traces_config import load_telemetry_config
from .telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    read_bounded_file,
    strict_json_loads,
)
from .telemetry_traces_report import (
    MAX_REPORT_BYTES,
    REPORT_FILE,
    REPORT_KIND,
    build_trace_export,
    build_trace_export_report,
    write_trace_export,
)
from .telemetry_traces_verify import verify_trace_export

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_journey.telemetry_traces_cli",
        description=(
            "OTLP/JSON traces and a Grafana dashboard derived from verified "
            "recovery-lab evidence; local CPU lab telemetry only."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="verify runs and export their traces", allow_abbrev=False
    )
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--serving-run-dir", type=Path)
    export.add_argument("--worker-run-dir", type=Path)
    export.add_argument("--output-dir", type=Path, required=True)
    dashboard = commands.add_parser(
        "dashboard", help="generate the Grafana dashboard", allow_abbrev=False
    )
    dashboard.add_argument("--config", type=Path, required=True)
    dashboard.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="recompute a stored export or dashboard", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--serving-run-dir", type=Path)
    verify.add_argument("--worker-run-dir", type=Path)
    verify.add_argument("--config", type=Path)
    return parser


def _new_output(path: Path) -> None:
    # Refuse a taken destination before spending time verifying the sources.
    if os.path.lexists(path):
        raise TelemetryError("output directory must not already exist")


def _export(arguments: argparse.Namespace) -> dict[str, Any]:
    runs = {
        "serving_run_dir": arguments.serving_run_dir,
        "worker_run_dir": arguments.worker_run_dir,
    }
    sources = [path for path in runs.values() if path is not None]
    if not sources:
        raise TelemetryError("export needs --serving-run-dir, --worker-run-dir or both")
    config = load_telemetry_config(arguments.config)
    _new_output(arguments.output_dir)
    export = build_trace_export(config, **runs)
    report = build_trace_export_report(config, export)
    write_trace_export(arguments.output_dir, export, report, sources=sources)
    return verify_trace_export(
        arguments.output_dir / REPORT_FILE, config_path=arguments.config, **runs
    )


def _dashboard(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_telemetry_config(arguments.config)
    _new_output(arguments.output_dir)
    artifact = build_dashboard_artifact(config)
    report = build_dashboard_report(config, artifact)
    write_dashboard_artifact(arguments.output_dir, artifact, report)
    return verify_dashboard_report(
        arguments.output_dir / DASHBOARD_REPORT_FILE, config_path=arguments.config
    )


def _report_kind(path: Path) -> object:
    data = read_bounded_file(path, max_bytes=MAX_REPORT_BYTES, label="report")
    try:
        report = strict_json_loads(data, label="report")
    except TelemetryError as error:
        raise TelemetryVerificationError(str(error)) from error
    return report.get("kind") if type(report) is dict else None


def _verify(arguments: argparse.Namespace) -> dict[str, Any]:
    kind = _report_kind(arguments.report)
    runs = (arguments.serving_run_dir, arguments.worker_run_dir)
    if kind == REPORT_KIND:
        return verify_trace_export(
            arguments.report,
            serving_run_dir=arguments.serving_run_dir,
            worker_run_dir=arguments.worker_run_dir,
            config_path=arguments.config,
        )
    if kind == DASHBOARD_REPORT_KIND:
        if any(path is not None for path in runs):
            raise TelemetryError("run directories apply only to trace export reports")
        return verify_dashboard_report(arguments.report, config_path=arguments.config)
    raise TelemetryVerificationError("report is neither a trace export nor a dashboard")


_COMMANDS = {"export": _export, "dashboard": _dashboard, "verify": _verify}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command, print its JSON receipt and return an explicit exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        receipt = _COMMANDS[arguments.command](arguments)
    except TelemetryVerificationError as error:
        print(f"telemetry {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"telemetry {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
