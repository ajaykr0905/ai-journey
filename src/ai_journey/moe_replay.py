"""Recompute every MoE experiment measurement from its supplied source bytes."""

from hashlib import sha256

import torch

from .moe_experiment_comparison import run_moe_document_comparison
from .moe_report import canonical_moe_json, verify_moe_report


def replay_moe_report(report, train_bytes, heldout_bytes):
    """Separate actual executable replay from offline internal consistency.

    Exact equality is checked in the declared float32 CPU protocol with the
    recorded PyTorch version. It is not a cross-platform reproducibility promise.
    No input artifact is modified and no accelerator RNG is seeded.
    """

    plan = verify_moe_report(report)
    if report["runtime"]["torch_version"] != str(torch.__version__):
        raise ValueError("MoE exact replay requires the recorded PyTorch version")
    if any(
        type(value) is not bytes or len(value) > 1024 * 1024
        for value in (train_bytes, heldout_bytes)
    ):
        raise ValueError("MoE replay requires bounded training and held-out bytes")
    trial = report["trials"][0]
    if (
        sha256(train_bytes).hexdigest() != trial["train_sha256"]
        or sha256(heldout_bytes).hexdigest() != trial["heldout_sha256"]
    ):
        raise ValueError("MoE replay source hashes differ from supplied bytes")
    expected = run_moe_document_comparison(plan, train_bytes, heldout_bytes)
    if canonical_moe_json(report) != canonical_moe_json(expected):
        raise ValueError("MoE report does not match exact executable replay")
    return plan
