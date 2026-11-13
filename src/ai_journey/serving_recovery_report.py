"""Canonical sealed report for one measured serving recovery experiment.

The report binds the config, source, model and every raw evidence file by
SHA-256, then records the fault, per-request reconciliation, measured timings
and the claim boundary. Its seal is the SHA-256 of the canonical report without
the seal field. A seal proves content identity, not authorship or that a timing
was honestly collected; ``serving_recovery_verify`` recomputes every derived
value from the raw evidence instead of trusting this summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    canonical_sha256,
    read_bounded_file,
    sha256_hex,
)
from .serving_recovery_experiment import (
    ATTEMPTS_FILE,
    CONFIG_FILE,
    SOURCE_FILE,
    TRACE_FILE,
    ServingRecoveryResult,
    journal_file,
)
from .serving_recovery_identity import lifetime_number
from .serving_recovery_io import (
    MAX_ATTEMPT_LOG_BYTES,
    MAX_TRACE_BYTES,
    load_canonical_json,
    publish_new_file,
)
from .serving_recovery_metrics import PERCENTILE_METHOD

REPORT_FILE = "report.json"
REPORT_KIND = "ai-journey-serving-recovery-report"
MAX_REPORT_BYTES = 512 * 1024
CLOCK = "time.monotonic_ns in the experiment process on one host"
EVIDENCE_BOUNDARY = (
    "Measured on one CPU host over 127.0.0.1 loopback: a supervisor SIGKILLed "
    "one local server process running a tiny deterministic character-bigram "
    "model built from a public names fixture, then restarted it. This is not "
    "GPU, CUDA, distributed, multi-host or production serving evidence and "
    "makes no model-quality, capacity or cost claim; timings depend on this "
    "host and its load."
)


def _milliseconds(value: int | None) -> float | None:
    return None if value is None else round(value / 1_000_000, 3)


def summary_milliseconds(timing: dict[str, Any]) -> dict[str, float | None]:
    """Human-readable durations derived from the exact nanosecond timing."""

    return {
        name.removesuffix("_ns") + "_ms": _milliseconds(timing[name])
        for name in (
            "detection_latency_ns",
            "restart_latency_ns",
            "unavailability_ns",
            "time_to_slo_recovery_ns",
        )
    }


def evidence_file_digest(run_dir: Path, name: str, max_bytes: int) -> str:
    return sha256_hex(read_bounded_file(run_dir / name, max_bytes=max_bytes, name=name))


def build_serving_report(result: ServingRecoveryResult, *, run_dir: Path) -> dict:
    """Seal identities, raw evidence digests, measurements and the boundary."""

    if not isinstance(result, ServingRecoveryResult):
        raise TypeError("result must be ServingRecoveryResult")
    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    config = result.config
    timing = result.timing.to_mapping()
    lifetimes = [
        event.lifetime_id
        for event in result.events
        if event.kind in ("spawn", "restart")
    ]
    journals = []
    for journal in result.journals:
        name = journal_file(lifetime_number(journal.lifetime_id))
        present = (run_dir / name).exists()
        journals.append(
            {
                "lifetime_id": journal.lifetime_id,
                "file": name if present else None,
                "file_sha256": (
                    evidence_file_digest(run_dir, name, MAX_ATTEMPT_LOG_BYTES)
                    if present
                    else None
                ),
                "records": len(journal.records),
                "torn_tail_bytes": journal.torn_tail_bytes,
            }
        )
    if [journal["lifetime_id"] for journal in journals] != lifetimes:
        raise ServingRecoveryError("journals do not cover every spawned lifetime")
    slo_recovered = result.timing.slo_recovered_ns is not None
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "experiment": {
            "run_id": result.run_id,
            "config": config.to_mapping(),
            "config_sha256": config.fingerprint(),
            "source_bytes": result.source_bytes,
            "source_sha256": result.source_sha256,
            "model_sha256": result.model_sha256,
            "lifetimes": lifetimes,
            "run_duration_ns": result.run_duration_ns,
        },
        "evidence": {
            "source_file": SOURCE_FILE,
            "config_file": CONFIG_FILE,
            "config_file_sha256": evidence_file_digest(run_dir, CONFIG_FILE, 16 * 1024),
            "attempts_file": ATTEMPTS_FILE,
            "attempts_file_sha256": evidence_file_digest(
                run_dir, ATTEMPTS_FILE, MAX_ATTEMPT_LOG_BYTES
            ),
            "trace_file": TRACE_FILE,
            "trace_file_sha256": evidence_file_digest(
                run_dir, TRACE_FILE, MAX_TRACE_BYTES
            ),
            "trace_head_sha256": result.trace_sha256,
            "journals": journals,
        },
        "fault": {
            "mechanism": "process_kill",
            "signal": result.fault.signal,
            "lifetime_id": result.fault.lifetime_id,
            "acknowledged_responses": result.fault.acknowledged_responses,
            "injected_ns": result.fault.injected_ns,
        },
        "reconciliation": result.reconciliation.to_mapping(),
        "timing": timing,
        "summary_ms": summary_milliseconds(timing),
        "slo": {
            "latency_ms": config.slo_latency_ms,
            "consecutive_successes": config.slo_consecutive_successes,
            "recovered": slo_recovered,
        },
        "recovered": slo_recovered and result.reconciliation.totals["failed"] == 0,
        "clock": CLOCK,
        "percentile_method": PERCENTILE_METHOD,
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }
    return seal_serving_report(report)


def seal_serving_report(report: dict) -> dict:
    """Add the canonical seal over every other report field."""

    if type(report) is not dict or "report_sha256" in report:
        raise ServingRecoveryError("report must be an unsealed mapping")
    return {**report, "report_sha256": canonical_sha256(report)}


def require_sealed_report(report: object) -> dict:
    """Return a report whose seal matches its canonical content."""

    if type(report) is not dict or type(report.get("report_sha256")) is not str:
        raise ServingRecoveryError("report is not sealed")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if canonical_sha256(unsigned) != report["report_sha256"]:
        raise ServingRecoveryError("report seal does not match its content")
    return report


def write_serving_report(
    path: Path, report: dict, *, protected: Sequence[Path] = ()
) -> None:
    """Publish one sealed canonical report without overwriting anything."""

    require_sealed_report(report)
    publish_new_file(
        path,
        canonical_json_bytes(report),
        max_bytes=MAX_REPORT_BYTES,
        protected=protected,
    )


def load_serving_report(path: Path) -> dict:
    """Load a bounded canonical report and check its seal."""

    return require_sealed_report(
        load_canonical_json(path, max_bytes=MAX_REPORT_BYTES, name="report")
    )
