"""End-to-end local worker crash, detection, restart, and recovery experiment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .recovery_checkpoint_io import load_recovery_checkpoint
from .recovery_config import (
    RecoveryExperimentConfig,
    RecoveryExperimentError,
    config_from_mapping,
)
from .recovery_detection import wait_for_process_exit, wait_for_progress_state
from .recovery_fault import inject_process_kill
from .recovery_identity import recovery_attempt_id, recovery_run_id
from .recovery_loss import RecoveryLossAssessment, assess_recovery_loss
from .recovery_metrics import RecoveryTiming, measure_recovery_timing
from .recovery_process import start_worker_process
from .recovery_resume import RecoveryResumeDecision, admit_recovery_resume
from .recovery_trace import RecoveryEvent
from .recovery_trace_io import append_recovery_event, load_recovery_trace
from .recovery_workload import RecoveryWorkload, build_recovery_workload


@dataclass(frozen=True)
class RecoveryExperimentResult:
    """Verified measurements and identities from one local process recovery."""

    config: RecoveryExperimentConfig
    workload: RecoveryWorkload
    run_id: str
    failed_attempt_id: str
    resumed_attempt_id: str
    failed_exit_code: int
    resumed_exit_code: int
    resume: RecoveryResumeDecision
    loss: RecoveryLossAssessment
    timing: RecoveryTiming
    trace_sha256: str


def _load_inputs(source_path: Path, config_path: Path):
    for name, path, bound in (
        ("source", source_path, 1024 * 1024),
        ("config", config_path, 64 * 1024),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > bound:
            raise RecoveryExperimentError(
                f"{name} must be a bounded regular non-symlink file"
            )
    try:
        config = config_from_mapping(json.loads(config_path.read_bytes()))
    except json.JSONDecodeError as exc:
        raise RecoveryExperimentError("config is not valid JSON") from exc
    return config, build_recovery_workload(source_path.read_bytes(), config)


def run_recovery_experiment(
    *, source_path: Path, config_path: Path, run_dir: Path
) -> RecoveryExperimentResult:
    """Kill a real worker process, restart it, and reconcile measured recovery."""

    if any(not isinstance(path, Path) for path in (source_path, config_path, run_dir)):
        raise TypeError("experiment paths must be pathlib.Path values")
    if run_dir.exists() or run_dir.is_symlink():
        raise RecoveryExperimentError("run_dir must not already exist")
    config, workload = _load_inputs(source_path, config_path)
    run_dir.mkdir(parents=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RecoveryExperimentError("run_dir must be a real directory")
    checkpoint_path = run_dir / "checkpoint.json"
    progress_path = run_dir / "progress.json"
    trace_path = run_dir / "trace.json"
    run_identity = recovery_run_id(config)
    first_attempt = recovery_attempt_id(run_identity, 1)
    started_ns = time.monotonic_ns()
    append_recovery_event(
        trace_path,
        RecoveryEvent(
            sequence=0,
            run_id=run_identity,
            attempt_id=first_attempt,
            kind="started",
            monotonic_ns=started_ns,
            completed_items=0,
            payload={"workload_sha256": workload.fingerprint()},
        ),
    )
    first = start_worker_process(
        source_path=source_path,
        config_path=config_path,
        run_id=run_identity,
        attempt_id=first_attempt,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        start_items=0,
        fault_after_items=config.fault_after_items,
    )
    second = None
    try:
        failed_progress = wait_for_progress_state(
            first,
            progress_path,
            "fault_ready",
            timeout_ms=config.worker_timeout_ms,
        )
        fault = inject_process_kill(first, failed_progress, config)
        append_recovery_event(
            trace_path,
            RecoveryEvent(
                sequence=1,
                run_id=run_identity,
                attempt_id=first_attempt,
                kind="fault_injected",
                monotonic_ns=fault.injected_monotonic_ns,
                completed_items=fault.completed_items,
                payload={"mechanism": fault.mechanism},
            ),
        )
        failed_exit_code, detected_ns = wait_for_process_exit(
            first, timeout_ms=config.heartbeat_timeout_ms
        )
        if failed_exit_code == 0:
            raise RecoveryExperimentError("faulted worker exited successfully")
        append_recovery_event(
            trace_path,
            RecoveryEvent(
                sequence=2,
                run_id=run_identity,
                attempt_id=first_attempt,
                kind="failure_detected",
                monotonic_ns=detected_ns,
                completed_items=failed_progress.completed_items,
                payload={"exit_code": failed_exit_code},
            ),
        )
        checkpoint = load_recovery_checkpoint(checkpoint_path)
        resume = admit_recovery_resume(checkpoint, failed_progress, workload, config)
        restart_started_ns = time.monotonic_ns()
        append_recovery_event(
            trace_path,
            RecoveryEvent(
                sequence=3,
                run_id=run_identity,
                attempt_id=resume.next_attempt_id,
                kind="resumed",
                monotonic_ns=restart_started_ns,
                completed_items=resume.start_items,
                payload={"parent_checkpoint_sha256": resume.parent_checkpoint_sha256},
            ),
        )
        second = start_worker_process(
            source_path=source_path,
            config_path=config_path,
            run_id=run_identity,
            attempt_id=resume.next_attempt_id,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            start_items=resume.start_items,
            fault_after_items=None,
        )
        final_progress = wait_for_progress_state(
            second,
            progress_path,
            "completed",
            timeout_ms=config.worker_timeout_ms,
        )
        resumed_exit_code, _ = wait_for_process_exit(
            second, timeout_ms=config.heartbeat_timeout_ms
        )
        if resumed_exit_code != 0:
            raise RecoveryExperimentError("resumed worker did not exit successfully")
        completed_ns = time.monotonic_ns()
        loss = assess_recovery_loss(
            failed_progress=failed_progress,
            recovery_checkpoint=checkpoint,
            final_progress=final_progress,
            workload=workload,
            config=config,
        )
        timing = measure_recovery_timing(
            fault_injected_ns=fault.injected_monotonic_ns,
            failure_detected_ns=detected_ns,
            restart_started_ns=restart_started_ns,
            recovery_completed_ns=completed_ns,
        )
        append_recovery_event(
            trace_path,
            RecoveryEvent(
                sequence=4,
                run_id=run_identity,
                attempt_id=resume.next_attempt_id,
                kind="completed",
                monotonic_ns=completed_ns,
                completed_items=final_progress.completed_items,
                payload={
                    "accumulator_sha256": final_progress.accumulator_sha256,
                    "recovery_time_ms": timing.recovery_time_ms,
                },
            ),
        )
        load_recovery_trace(trace_path)
        return RecoveryExperimentResult(
            config=config,
            workload=workload,
            run_id=run_identity,
            failed_attempt_id=first_attempt,
            resumed_attempt_id=resume.next_attempt_id,
            failed_exit_code=failed_exit_code,
            resumed_exit_code=resumed_exit_code,
            resume=resume,
            loss=loss,
            timing=timing,
            trace_sha256=json.loads(trace_path.read_bytes())[-1]["event_sha256"],
        )
    finally:
        if first.poll() is None:
            first.kill()
        if second is not None and second.poll() is None:
            second.kill()
