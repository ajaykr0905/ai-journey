"""Canonical public-safe report for measured worker recovery experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from typing import Any

from .recovery_experiment import RecoveryExperimentResult


def build_recovery_report(result: RecoveryExperimentResult) -> dict[str, Any]:
    """Seal identities, actual timings, work loss, and the claim boundary."""

    if not isinstance(result, RecoveryExperimentResult):
        raise TypeError("result must be RecoveryExperimentResult")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": {
            "run_id": result.run_id,
            "config_sha256": result.config.fingerprint(),
            "workload_sha256": result.workload.fingerprint(),
            "source_bytes": result.workload.source_bytes,
            "source_sha256": result.workload.source_sha256,
            "work_items": len(result.workload.items),
        },
        "attempts": {
            "failed_attempt_id": result.failed_attempt_id,
            "resumed_attempt_id": result.resumed_attempt_id,
            "failed_exit_code": result.failed_exit_code,
            "resumed_exit_code": result.resumed_exit_code,
            "parent_checkpoint_sha256": result.resume.parent_checkpoint_sha256,
        },
        "fault": {
            "mechanism": "process_kill",
            "fault_after_items": result.config.fault_after_items,
            "checkpoint_interval": result.config.checkpoint_interval,
        },
        "loss_assessment": asdict(result.loss),
        "timing": asdict(result.timing),
        "trace_sha256": result.trace_sha256,
        "evidence_boundary": (
            "Measured local CPU process-kill recovery on one host. This does not prove "
            "NVIDIA GPU, distributed, serving, production, model-quality, cost, or "
            "learner-completion evidence."
        ),
    }
    payload["report_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload
