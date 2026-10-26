"""Probe, plan and verify NVIDIA operational preparation without launching it."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .gpu_preflight import GpuTopology
from .gpu_preflight_evidence import (
    build_gpu_run_evidence,
    verify_gpu_rank_evidence,
    verify_gpu_run_evidence,
    validate_run_id,
)
from .gpu_preflight_io import read_gpu_json, write_gpu_json
from .gpu_preflight_launch import build_gpu_launch_plan
from .gpu_preflight_probe import probe_gpu_host


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    for name in ("plan", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--resource-report", type=Path, required=True)
        command.add_argument("--devices", type=int, nargs="+", required=True)
        command.add_argument(
            "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
        )
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--run-id", required=name == "collect")
        command.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            payload = probe_gpu_host()
        elif args.command == "verify":
            payload = read_gpu_json(args.report)
            if payload.get("kind") == "nvidia_rank_preflight":
                verify_gpu_rank_evidence(payload)
            else:
                verify_gpu_run_evidence(payload)
        else:
            topology = GpuTopology(tuple(args.devices), dtype=args.dtype)
            run_id = args.run_id or str(uuid4())
            validate_run_id(run_id)
            if args.output.resolve() == args.resource_report.resolve():
                raise ValueError(
                    "preflight output must not overwrite its resource input"
                )
            rank_paths = [
                args.output_dir / run_id / f"rank-{rank}.json"
                for rank in range(topology.world_size)
            ]
            if args.output.resolve() in [path.resolve() for path in rank_paths]:
                raise ValueError("preflight summary must not overwrite a rank record")
            if args.command == "plan":
                payload = build_gpu_launch_plan(
                    resource_report_path=args.resource_report,
                    topology=topology,
                    output_dir=args.output_dir,
                    run_id=run_id,
                    probe=probe_gpu_host(),
                )
            else:
                payload = build_gpu_run_evidence(
                    [read_gpu_json(path) for path in rank_paths],
                    run_id=run_id,
                    topology=topology,
                    resource_report=read_gpu_json(args.resource_report),
                )
        if args.command != "verify":
            write_gpu_json(args.output, payload)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
