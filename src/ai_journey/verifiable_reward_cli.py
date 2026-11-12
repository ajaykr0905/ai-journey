"""Build, verify, and replay deterministic verifiable-reward evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from .verifiable_reward_experiment import run_reward_experiment
from .verifiable_reward_faults import FaultPlan, InjectedFault
from .verifiable_reward_group import CandidateGroup, admit_candidate_group
from .verifiable_reward_io import load_reward_report, write_reward_report
from .verifiable_reward_replay import replay_reward_report
from .verifiable_reward_report import build_reward_report
from .verifiable_reward_task import VerifiableTask
from .verifiable_reward_verify import verify_reward_report


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise VerifiableRewardError(f"duplicate config field: {key}")
        value[key] = item
    return value


def load_reward_fixture(
    path: Path,
) -> tuple[
    tuple[VerifiableTask, ...],
    tuple[CandidateGroup, ...],
    RewardRunConfig,
    FaultPlan,
]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise VerifiableRewardError("config must be a bounded regular non-symlink file")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise VerifiableRewardError("config is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "run",
        "tasks",
        "groups",
        "faults",
    }:
        raise VerifiableRewardError("config root fields mismatch")
    if not isinstance(value["run"], dict):
        raise VerifiableRewardError("run must be an object")
    config = RewardRunConfig.from_mapping(value["run"])
    if not isinstance(value["tasks"], list) or not all(
        isinstance(item, dict) for item in value["tasks"]
    ):
        raise VerifiableRewardError("tasks must be a list of objects")
    tasks = tuple(VerifiableTask.from_mapping(item) for item in value["tasks"])
    if not isinstance(value["groups"], list):
        raise VerifiableRewardError("groups must be a list")
    groups = []
    for item in value["groups"]:
        if not isinstance(item, dict) or set(item) != {"task_id", "responses"}:
            raise VerifiableRewardError("group fields mismatch")
        if not isinstance(item["task_id"], str) or not isinstance(
            item["responses"], list
        ):
            raise VerifiableRewardError("group values have invalid types")
        groups.append(admit_candidate_group(item["task_id"], item["responses"], config))
    if not isinstance(value["faults"], list):
        raise VerifiableRewardError("faults must be a list")
    faults = []
    for item in value["faults"]:
        if not isinstance(item, dict) or set(item) != {
            "task_id",
            "candidate_index",
            "attempt",
            "kind",
        }:
            raise VerifiableRewardError("fault fields mismatch")
        if (
            not isinstance(item["task_id"], str)
            or isinstance(item["candidate_index"], bool)
            or not isinstance(item["candidate_index"], int)
            or isinstance(item["attempt"], bool)
            or not isinstance(item["attempt"], int)
            or not isinstance(item["kind"], str)
        ):
            raise VerifiableRewardError("fault values have invalid types")
        faults.append(InjectedFault(**item))
    return tasks, tuple(groups), config, FaultPlan(faults)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="evaluate fixtures and seal a report")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    for name, help_text in (
        ("verify", "independently reconstruct a sealed report"),
        ("replay", "rerun a sealed report byte exactly"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--report", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        tasks, groups, config, faults = load_reward_fixture(config_path)
        if args.command == "build":
            output = args.output.resolve()
            if output == config_path:
                raise VerifiableRewardError("output must preserve the config input")
            report = build_reward_report(
                tasks,
                groups,
                config,
                faults,
                run_reward_experiment(tasks, groups, config, faults),
            )
            write_reward_report(output, report)
            summary = {
                "report_sha256": report["sha256"],
                "candidate_count": report["result"]["metrics"]["candidate_count"],
                "correct_count": report["result"]["metrics"]["correct_count"],
            }
        else:
            report = load_reward_report(args.report.resolve())
            summary = (
                verify_reward_report(report, tasks, groups, config, faults)
                if args.command == "verify"
                else replay_reward_report(report, tasks, groups, config, faults)
            )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
