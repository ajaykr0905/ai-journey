"""Exercise installed NVIDIA preflight commands as a CPU package-integrity gate.

Workers run sequentially with explicit single-node rank environments. This gate
does not execute the planned torchrun command, CUDA kernels or NCCL collectives.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _text(value):
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else value or ""
    )


def _run_command(command, *, name, directory, environment, expected=(0,), timeout=30):
    """Keep diagnostic streams even when a bounded command fails."""
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        (directory / f"{name}.stdout.txt").write_text(_text(error.stdout))
        (directory / f"{name}.stderr.txt").write_text(_text(error.stderr))
        raise RuntimeError(f"{name} exceeded its readiness deadline") from error
    (directory / f"{name}.stdout.txt").write_text(result.stdout)
    (directory / f"{name}.stderr.txt").write_text(result.stderr)
    if result.returncode not in expected:
        raise RuntimeError(
            f"{name} exited {result.returncode}; inspect retained command logs"
        )
    return {"name": name, "returncode": result.returncode}


def run_readiness(
    *,
    installed_root: Path,
    resource_config: Path,
    output_dir: Path,
    expect_blocked=False,
):
    """Prove wheel entry points can collect and verify real blocked observations."""
    installed_root = installed_root.resolve()
    if not Path(__file__).resolve().is_relative_to(installed_root):
        raise ValueError(
            "readiness must import the installed wheel, not the source checkout"
        )
    commands = {
        name: installed_root / "bin" / name
        for name in ("ai-journey-resource-plan", "ai-journey-gpu-preflight")
    }
    if any(
        not path.is_file() or not path.resolve().is_relative_to(installed_root)
        for path in commands.values()
    ):
        raise ValueError(
            "installed console entry points are missing from the isolated target"
        )
    resource_config = resource_config.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "PYTHONPATH": str(installed_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = []

    def run(arguments, name, *, worker_environment=None, expected=(0,)):
        completed.append(
            _run_command(
                arguments,
                name=name,
                directory=output_dir,
                environment=environment
                if worker_environment is None
                else worker_environment,
                expected=expected,
            )
        )

    resource_path = output_dir / "resources.json"
    run(
        [
            str(commands["ai-journey-resource-plan"]),
            "--config",
            str(resource_config),
            "--output",
            str(resource_path),
        ],
        "resource-plan",
    )
    resource = json.loads(resource_path.read_text())
    world_size = resource["workload"]["data_parallel_size"]
    if type(world_size) is not int or not 1 <= world_size <= 8:
        raise ValueError("CPU package gate requires one to eight rank observations")
    devices = [str(rank) for rank in range(world_size)]
    dtype = resource["assumptions"]["parameter_dtype"]
    gpu_command = str(commands["ai-journey-gpu-preflight"])
    probe_path = output_dir / "host-probe.json"
    run([gpu_command, "probe", "--output", str(probe_path)], "probe")
    probe = json.loads(probe_path.read_text())
    if expect_blocked and probe["available"]:
        raise RuntimeError("CPU integrity gate expected an unavailable NVIDIA host")
    rank_dir = output_dir / "ranks"
    plan_path = output_dir / "launch-plan.json"
    common = [
        "--resource-report",
        str(resource_path),
        "--devices",
        *devices,
        "--dtype",
        dtype,
        "--output-dir",
        str(rank_dir),
    ]
    run([gpu_command, "plan", *common, "--output", str(plan_path)], "plan")
    plan = json.loads(plan_path.read_text())
    if plan["command_executed"] is not False or plan["training_executed"] is not False:
        raise RuntimeError("launch planning must retain its non-execution contract")
    run_id = plan["run_id"]
    for rank in range(world_size):
        worker_environment = {
            **environment,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
        }
        run(
            [
                sys.executable,
                "-m",
                "ai_journey.gpu_preflight_worker",
                *common,
                "--run-id",
                run_id,
            ],
            f"rank-{rank}",
            worker_environment=worker_environment,
            expected=(0, 2),
        )
        record = json.loads((rank_dir / run_id / f"rank-{rank}.json").read_text())
        if (completed[-1]["returncode"] == 0) != record["admission"]["preflight_ready"]:
            raise RuntimeError("worker exit status does not match observed admission")
    evidence_path = output_dir / "run-evidence.json"
    run(
        [
            gpu_command,
            "collect",
            *common,
            "--run-id",
            run_id,
            "--output",
            str(evidence_path),
        ],
        "collect",
    )
    run([gpu_command, "verify", "--report", str(evidence_path)], "verify-run")
    run(
        [gpu_command, "verify", "--report", str(rank_dir / run_id / "rank-0.json")],
        "verify-rank",
    )
    evidence = json.loads(evidence_path.read_text())
    if expect_blocked and evidence["preflight_passed"]:
        raise RuntimeError("blocked CPU observations cannot pass NVIDIA admission")
    for key in ("training_executed", "gpu_kernel_executed", "collective_executed"):
        if evidence[key] is not False:
            raise RuntimeError("CPU package gate cannot certify GPU execution")
    summary = {
        "schema_version": 1,
        "kind": "nvidia_installed_package_integrity_gate",
        "integrity_smoke_passed": True,
        "hardware_preflight_passed": evidence["preflight_passed"],
        "host_available": probe["available"],
        "host_reason": probe["reason"],
        "run_id": run_id,
        "world_size": world_size,
        "worker_launcher": "sequential_processes_with_explicit_rank_environment",
        "torchrun_executed": False,
        "training_executed": False,
        "gpu_kernel_executed": False,
        "collective_executed": False,
        "installed_module": str(Path(__file__).resolve().relative_to(installed_root)),
        "commands": completed,
    }
    (output_dir / "readiness.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--resource-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-blocked", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_readiness(
            installed_root=args.installed_root,
            resource_config=args.resource_config,
            output_dir=args.output_dir,
            expect_blocked=args.expect_blocked,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
