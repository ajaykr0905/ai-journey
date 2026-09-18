#!/usr/bin/env python3
"""Validate deployment prerequisites from a small JSON configuration.

The checker is dependency-free and safe for CI. It checks required files,
directories, executables, and environment-variable names without ever exposing
environment values. It prints a terminal report and can also write JSON.

Exit codes:

* 0: every required check passed;
* 1: one or more required checks failed;
* 2: the configuration is invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, list[str]] = {
    "required_files": [
        "README.md",
        "PROGRESS.md",
        "SECURITY.md",
        "requirements.txt",
        "scripts/run_days_00_11.py",
        "scripts/run_day_12.py",
        "scripts/run_day_13.py",
        "scripts/run_day_14.py",
    ],
    "required_directories": ["days", "src/ai_journey", "tests"],
    "required_commands": ["python3"],
    "required_env": [],
    "optional_env": [],
}


@dataclass(frozen=True)
class CheckResult:
    """Normalized result for one readiness check."""

    category: str
    target: str
    status: str
    message: str


class ConfigurationError(ValueError):
    """Raised when the readiness configuration is malformed."""


class DeploymentReadinessChecker:
    """Evaluate configured deployment prerequisites."""

    def __init__(self, root: Path, config: dict[str, list[str]]) -> None:
        self.root = root.resolve()
        self.config = config

    def run(self) -> list[CheckResult]:
        """Run every configured check in a stable order."""

        results: list[CheckResult] = []
        results.extend(self._check_paths("file", "required_files", is_file=True))
        results.extend(
            self._check_paths("directory", "required_directories", is_file=False)
        )
        results.extend(self._check_commands())
        results.extend(self._check_environment("required_env", "fail"))
        results.extend(self._check_environment("optional_env", "warn"))
        return results

    def _check_paths(
        self, category: str, config_key: str, *, is_file: bool
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        for target in self.config[config_key]:
            path = self._resolve_path(target)
            found = path.is_file() if is_file else path.is_dir()
            expected = "file" if is_file else "directory"
            if found:
                results.append(
                    CheckResult(category, target, "pass", f"Found {expected}")
                )
            else:
                results.append(
                    CheckResult(
                        category,
                        target,
                        "fail",
                        f"Missing {expected} at {path}",
                    )
                )
        return results

    def _check_commands(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for command in self.config["required_commands"]:
            executable = shutil.which(command)
            if executable:
                results.append(
                    CheckResult(
                        "command", command, "pass", f"Available at {executable}"
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "command", command, "fail", "Executable not found on PATH"
                    )
                )
        return results

    def _check_environment(
        self, config_key: str, missing_status: str
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        for variable in self.config[config_key]:
            if os.environ.get(variable):
                results.append(
                    CheckResult("environment", variable, "pass", "Variable is set")
                )
                continue

            message = (
                "Optional variable is not set"
                if missing_status == "warn"
                else "Required variable is not set"
            )
            results.append(
                CheckResult("environment", variable, missing_status, message)
            )
        return results

    def _resolve_path(self, target: str) -> Path:
        path = Path(target)
        if path.is_absolute():
            raise ConfigurationError(
                f"Paths must be relative to the readiness root: {target}"
            )
        return self.root / path


def load_config(path: Path | None = None) -> dict[str, list[str]]:
    """Load a JSON configuration and validate its known fields."""

    loaded_config: dict[str, Any] = {}
    if path:
        try:
            with path.open(encoding="utf-8") as config_file:
                loaded = json.load(config_file)
        except FileNotFoundError as exc:
            raise ConfigurationError(f"Configuration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError("Configuration must be a JSON object")
        loaded_config = loaded

    unknown_keys = set(loaded_config) - set(DEFAULT_CONFIG)
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ConfigurationError(f"Unknown configuration keys: {names}")

    config = {**DEFAULT_CONFIG, **loaded_config}
    for key in DEFAULT_CONFIG:
        value = config[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ConfigurationError(f"'{key}' must be a list of strings")
    return config


def build_report(root: Path, results: list[CheckResult]) -> dict[str, Any]:
    """Build a JSON-serializable report without exposing environment values."""

    summary = {"pass": 0, "warn": 0, "fail": 0}
    for result in results:
        summary[result.status] += 1
    return {
        "root": str(root.resolve()),
        "summary": summary,
        "ready": summary["fail"] == 0,
        "checks": [asdict(result) for result in results],
    }


def check_readiness(root: Path, config: dict[str, list[str]]) -> dict[str, Any]:
    """Run all checks and return the normalized report."""

    return build_report(root, DeploymentReadinessChecker(root, config).run())


def print_report(report: dict[str, Any]) -> None:
    """Print the report in a format readable in CI logs."""

    for check in report["checks"]:
        print(
            f"[{check['status'].upper():4}] {check['category']:<11} "
            f"{check['target']}: {check['message']}"
        )

    summary = report["summary"]
    state = "READY" if report["ready"] else "NOT READY"
    print(
        f"\n{state} - {summary['pass']} passed, "
        f"{summary['warn']} warnings, {summary['fail']} failed"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory to validate (default: repository root)",
    )
    parser.add_argument(
        "--config", type=Path, help="JSON file containing readiness requirements"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the checker and return a CI-friendly status code."""

    args = parse_args(argv)
    try:
        config = load_config(args.config)
        report = check_readiness(args.root, config)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
