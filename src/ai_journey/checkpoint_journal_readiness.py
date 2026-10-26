"""Prove installed two-rank journal fallback using a private CPU fixture run."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from .ddp_acceptance import exact_state


def run_readiness(*, installed_root: Path, repository: Path, output_dir: Path):
    """Keep all proof artifacts; never corrupt an existing user's checkpoint."""
    installed_root = installed_root.resolve()
    if not Path(__file__).resolve().is_relative_to(installed_root):
        raise ValueError("journal readiness requires an isolated installed package")
    command = installed_root / "bin/ai-journey-ddp"
    if not command.is_file() or not command.resolve().is_relative_to(installed_root):
        raise ValueError("isolated installed DDP entry point is missing")
    repository = repository.resolve()
    source = repository / "data/day-19-demo-names.txt"
    spec = repository / "config/ddp-lab.json"
    if not source.is_file() or not spec.is_file():
        raise ValueError(
            "the checked public CPU fixture and specification are required"
        )
    if output_dir.is_symlink():
        raise ValueError("readiness output must not be a symlink")
    output_dir = output_dir.absolute()
    if any(path.is_relative_to(output_dir.resolve()) for path in (source, spec)):
        raise ValueError("readiness output must be disjoint from protected inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "PYTHONPATH": str(installed_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    journal = output_dir / "versions"

    def train(name, steps, *, versioned=False, resume=False):
        checkpoint = output_dir / ("recovered.pt" if versioned else "continuous.pt")
        arguments = [
            str(command),
            "train",
            "--source",
            str(source),
            "--spec",
            str(spec),
            "--checkpoint",
            str(checkpoint),
            "--workdir",
            str(output_dir / name),
            "--output",
            str(output_dir / f"{name}.json"),
            "--world-size",
            "2",
            "--steps",
            str(steps),
        ]
        if versioned:
            arguments += [
                "--journal-directory",
                str(journal),
                "--checkpoint-interval",
                "1",
            ]
        if resume:
            arguments.append("--resume")
        # The command also enforces bounded worker deadlines and reaps its children.
        result = subprocess.run(
            arguments,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        (output_dir / f"{name}.stdout.txt").write_text(result.stdout)
        (output_dir / f"{name}.stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"{name} failed; inspect retained journal readiness logs"
            )
        return json.loads((output_dir / f"{name}.json").read_bytes())

    baseline = train("continuous", 4)
    train("initial", 2, versioned=True)
    manifest = journal / "manifest.json"
    entries = json.loads(manifest.read_bytes())["entries"]
    selected = entries[0]["sha256"]
    # Only these brand-new, exclusively owned test artifacts are damaged.
    (journal / entries[-1]["filename"]).write_bytes(
        b"deliberate private fixture corruption"
    )
    (output_dir / "recovered.pt").write_bytes(b"latest file is not journal authority")
    resumed = train("resumed", 3, versioned=True, resume=True)
    if resumed["metrics"] != baseline["metrics"][1:]:
        raise ValueError("journal replay metrics do not match uninterrupted training")
    actual = torch.load(output_dir / "recovered.pt", weights_only=True)
    expected = torch.load(output_dir / "continuous.pt", weights_only=True)
    if not all(
        exact_state(actual[key], expected[key])
        for key in ("model", "optimizer", "rank_rngs", "step")
    ):
        raise ValueError("journal restart state does not exactly match the control")
    recovery = {"restored_step": 1, "recorded_step": 2, "checkpoint_sha256": selected}
    for rank in (0, 1):
        receipt = json.loads(
            (output_dir / "resumed" / f"rank-{rank}.json").read_bytes()
        )
        if receipt["journal_recovery"] != recovery:
            raise ValueError("rank did not restore the expected older journal version")
    retained = [entry["step"] for entry in json.loads(manifest.read_bytes())["entries"]]
    if retained != [1, 2, 3, 4]:
        raise ValueError(
            "journal history was rewritten or did not advance past high water"
        )
    report = {
        "installed_package": str(Path(__file__).resolve()),
        "world_size": 2,
        "restored_step": 1,
        "recorded_step": 2,
        "completed_step": 4,
        "model_sha256": resumed["model_sha256"],
        "selected_checkpoint_sha256": selected,
        "retained_steps": retained,
        "exact_model_optimizer_rank_rng_metrics": True,
        "scope": "local CPU/Gloo fixture; not GPU, scale, production or learner evidence",
    }
    (output_dir / "readiness.json").write_text(
        json.dumps(report, sort_keys=True) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_readiness(**vars(arguments)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
