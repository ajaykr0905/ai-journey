"""Operate an isolated installed replicated MoE lab and prove exact restart."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess

from .checkpoint_archive import load_checkpoint_archive, MAX_CHECKPOINT_RECORDS
from .ddp_acceptance import exact_state


def _command(arguments, *, name, directory, environment):
    # A separate, explicitly owned process group also contains the CLI's ranks.
    # If its own worker supervisor hangs, the outer deadline cannot orphan them.
    process = subprocess.Popen(
        arguments,
        cwd=directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    expired = False
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        expired = True
        # The inner launcher has already exceeded its own graceful worker
        # deadline. Kill the entire owned group before reaping its leader:
        # leader exit alone must not leave a TERM-ignoring rank running, and
        # keeping the leader unreaped avoids signaling a reused process ID.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=5)
    (directory / f"{name}.stdout.txt").write_text(stdout)
    (directory / f"{name}.stderr.txt").write_text(stderr)
    if expired or process.returncode:
        raise RuntimeError(f"{name} failed; inspect retained installed MoE logs")


def run_readiness(*, installed_root: Path, repository: Path, output_dir: Path):
    installed_root = installed_root.resolve()
    if not Path(__file__).resolve().is_relative_to(installed_root):
        raise ValueError("MoE readiness requires the isolated installed package")
    command = installed_root / "bin/ai-journey-moe-ddp"
    if not command.is_file() or not command.resolve().is_relative_to(installed_root):
        raise ValueError("the isolated installed MoE DDP entry point is required")
    repository = repository.resolve()
    source = repository / "data/day-19-demo-names.txt"
    config = repository / "config/moe-ddp-cpu.json"
    if not source.is_file() or not config.is_file():
        raise ValueError("the checked public MoE source and plan are required")
    if output_dir.is_symlink():
        raise ValueError("MoE readiness output cannot be a symlink")
    output_dir = output_dir.absolute()
    if any(path.is_relative_to(output_dir.resolve()) for path in (source, config)):
        raise ValueError("readiness output must be disjoint from protected inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "PYTHONPATH": str(installed_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def invoke(arguments, name):
        _command(
            [str(command), *map(str, arguments)],
            name=name,
            directory=output_dir,
            environment=environment,
        )

    invoke(["plan", "--config", config, "--source", source], "plan")

    def train(name, steps, *, resume=False):
        checkpoint = output_dir / (
            "continuous.pt" if name == "continuous" else "recovered.pt"
        )
        arguments = [
            "train",
            "--config",
            config,
            "--source",
            source,
            "--checkpoint",
            checkpoint,
            "--workdir",
            output_dir / name,
            "--output",
            output_dir / f"{name}.json",
            "--steps",
            steps,
        ]
        if resume:
            arguments.append("--resume")
        invoke(arguments, name)
        invoke(
            [
                "verify",
                "--report",
                output_dir / f"{name}.json",
                "--checkpoint",
                checkpoint,
            ],
            f"{name}-verify",
        )

    train("continuous", 4)
    train("initial", 2)
    train("resumed", 2, resume=True)
    expected = load_checkpoint_archive(
        output_dir / "continuous.pt",
        max_bytes=64 * 1024 * 1024,
        max_records=MAX_CHECKPOINT_RECORDS,
    )
    actual = load_checkpoint_archive(
        output_dir / "recovered.pt",
        max_bytes=64 * 1024 * 1024,
        max_records=MAX_CHECKPOINT_RECORDS,
    )
    if not all(
        exact_state(actual[key], expected[key])
        for key in (
            "inner",
            "rank_rngs",
            "step",
            "data_cursor",
            "data_contract",
            "config",
            "source_sha256",
            "world_size",
        )
    ):
        raise ValueError(
            "installed MoE restart state differs from uninterrupted training"
        )
    initial_pids, resumed_pids = [], []
    for rank in (0, 1):
        receipts = {
            name: json.loads((output_dir / name / f"rank-{rank}.json").read_bytes())
            for name in ("continuous", "initial", "resumed")
        }
        continuous, initial, resumed = (
            receipts[name] for name in ("continuous", "initial", "resumed")
        )
        if (
            initial["metrics"] + resumed["metrics"] != continuous["metrics"]
            or resumed["started_step"] != 2
            or resumed["completed_step"] != 4
            or initial["completed_step"] != 2
        ):
            raise ValueError("installed MoE replay metrics or progress differ")
        initial_pids.append(initial["pid"])
        resumed_pids.append(resumed["pid"])
    if set(initial_pids) & set(resumed_pids):
        raise ValueError(
            "installed MoE proof did not observe fresh resumed worker PIDs"
        )
    proof = {
        "installed_module": str(Path(__file__).resolve()),
        "world_size": actual["world_size"],
        "completed_step": actual["step"],
        "model_sha256": actual["inner"]["model_fingerprint"],
        "checkpoint_payload_sha256": actual["checkpoint_sha256"],
        "checkpoint_archive_sha256": sha256(
            (output_dir / "recovered.pt").read_bytes()
        ).hexdigest(),
        "initial_worker_pids": initial_pids,
        "resumed_worker_pids": resumed_pids,
        "exact_model_optimizer_rank_rng_data_cursor_metrics": True,
        "scope": "actual bounded CPU/Gloo restart; not GPU, expert parallelism, scale or learner evidence",
    }
    (output_dir / "readiness.json").write_text(json.dumps(proof, sort_keys=True) + "\n")
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run_readiness(**vars(parser.parse_args(argv))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
