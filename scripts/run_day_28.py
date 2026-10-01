#!/usr/bin/env python3
"""Verify manual cross-entropy against independent CPU gradient checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    from ai_journey.cross_entropy_audit import audit_cross_entropy
    from ai_journey.transformer_gradient_audit import audit_transformer_loss_gradient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--transformer-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    rng = np.random.default_rng(args.seed)
    cases = {
        "uniform_ties": (np.zeros((2, 3)), [0, 2]),
        "seeded_batch": (rng.normal(size=(4, 5)), rng.integers(5, size=4)),
        "underflowing_target": (np.array([[1000, -1000], [-1000, 1000]]), [1, 0]),
    }
    try:
        audits = {
            name: audit_cross_entropy(
                logits, labels, epsilon=args.epsilon, tolerance=args.tolerance
            ).to_dict()
            for name, (logits, labels) in cases.items()
        }
        transformer = audit_transformer_loss_gradient(
            seed=args.seed, tolerance=args.transformer_tolerance
        ).to_dict()
    except ValueError as exc:
        parser.error(str(exc))
    passed = all(audit["passed"] for audit in audits.values()) and transformer["passed"]
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "device": "cpu",
            "dtype": "float64",
        },
        "cases": audits,
        "transformer": transformer,
        "passed": passed,
        "limitations": [
            "Synthetic token batch; not a public-corpus training run.",
            "Only the loss derivative is manual; model derivatives use PyTorch.",
            "CPU diagnostic; no GPU, distributed-scale, or production claim.",
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
    print(
        f"passed={passed} cases={len(audits)} parameters={len(transformer['parameter_errors'])}"
    )
    print(f"report={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
