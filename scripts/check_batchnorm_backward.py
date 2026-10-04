"""Run deterministic manual BatchNorm backward checks on CPU float64."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    from ai_journey.batchnorm_backward import audit_batchnorm_backward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(30)
    x = rng.normal(size=(2, 3, 4))
    scale = np.array([0.0, -2.0, 0.5, 3.0])
    bias = rng.normal(size=4)
    upstream = rng.normal(size=x.shape)
    cases = {
        "sequence_batch": (x, scale, bias, upstream),
        "two_samples": (x[0, :2], scale, bias, upstream[0, :2]),
    }
    try:
        audits = {
            name: audit_batchnorm_backward(
                *values, epsilon=args.epsilon, tolerance=args.tolerance
            )
            for name, values in cases.items()
        }
    except (ValueError, FloatingPointError) as exc:
        parser.error(str(exc))
    passed = all(audit["passed"] for audit in audits.values())
    report = {
        "schema_version": 1,
        "seed": 30,
        "device": "cpu",
        "dtype": "float64",
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "epsilon": args.epsilon,
        "cases": audits,
        "passed": passed,
        "limitations": [
            "Training-mode biased batch variance only; no running-state update.",
            "Small synthetic diagnostic; no GPU, model-quality or learner claim.",
        ],
    }
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"passed={passed} cases={len(audits)} report={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
