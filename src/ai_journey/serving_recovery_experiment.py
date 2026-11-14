"""Compose one loopback serving kill-and-recover experiment end to end.

The runner loads strict controls and a public source, builds the model and the
run identities, and copies the source and canonical config into a new run
directory. It launches lifetime 1, drives bounded client load whose ledger
SIGKILLs the server after K acknowledged responses, lets the supervisor detect
the exit and start a replacement, waits until every logical request has
finished, stops the replacement with SIGTERM, then persists the raw evidence:
classified attempts, each lifetime's execution journal and the chained
lifecycle trace. Every child it spawned is reaped, including on failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .serving_recovery_client import AttemptRecord, PlannedRequest, plan_requests
from .serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    canonical_json_bytes,
    load_serving_recovery_config,
    read_bounded_file,
    sha256_hex,
)
from .serving_recovery_fault import AcknowledgedResponseFault, ServingFaultInjection
from .serving_recovery_identity import lifetime_number, serving_run_id
from .serving_recovery_io import publish_new_file, write_attempt_log, write_trace
from .serving_recovery_journal import JournalContents, read_execution_journal
from .serving_recovery_load import AttemptLedger, run_load
from .serving_recovery_metrics import ServingRecoveryTiming, measure_serving_recovery
from .serving_recovery_model import MAX_SOURCE_BYTES, build_serving_model
from .serving_recovery_reconcile import ServingReconciliation, reconcile_serving_run
from .serving_recovery_server import server_command
from .serving_recovery_supervisor import LifecycleObservation, ServingSupervisor
from .serving_recovery_trace import ServingLifecycleEvent, chain_serving_trace

SOURCE_FILE = "source.txt"
CONFIG_FILE = "config.json"
ATTEMPTS_FILE = "attempts.json"
TRACE_FILE = "trace.json"
# Lifecycle order for events that share one monotonic timestamp.
_TIE_ORDER = {
    "spawn": 0,
    "restart": 0,
    "ready": 1,
    "fault": 2,
    "unresponsive": 2,
    "exit_observed": 3,
    "recovered": 4,
    "exhausted": 5,
    "finished": 5,
}


def journal_file(number: int) -> str:
    return f"journal-lifetime-{number}.jsonl"


def journal_capacity(config: ServingRecoveryConfig) -> int:
    """Executions one lifetime may journal: every attempt of every request."""

    return config.requests * (config.retry_budget + 1)


@dataclass(frozen=True)
class ServingRecoveryResult:
    """Raw evidence and derived measurements of one completed experiment."""

    config: ServingRecoveryConfig
    run_id: str
    source_bytes: int
    source_sha256: str
    model_sha256: str
    plan: tuple[PlannedRequest, ...]
    attempts: tuple[AttemptRecord, ...]
    journals: tuple[JournalContents, ...]
    fault: ServingFaultInjection
    events: tuple[ServingLifecycleEvent, ...]
    trace_sha256: str
    reconciliation: ServingReconciliation
    timing: ServingRecoveryTiming
    run_duration_ns: int


def lifecycle_events(
    *,
    run_id: str,
    observations: tuple[LifecycleObservation, ...],
    fault: ServingFaultInjection,
    timing: ServingRecoveryTiming,
    config: ServingRecoveryConfig,
) -> tuple[ServingLifecycleEvent, ...]:
    """Merge supervisor, fault and SLO observations into one ordered trace."""

    rows = []
    parents: dict[int, str] = {}
    for order, item in enumerate(observations):
        parents.setdefault(item.lifetime_number, item.lifetime_id)
        if item.kind in ("spawn", "ready", "unresponsive"):
            kind, payload = item.kind, {"pid": item.pid}
            if item.kind == "ready":
                payload["port"] = item.port
        elif item.kind in ("exit_observed", "stopped"):
            kind = "exit_observed" if item.kind == "exit_observed" else "finished"
            payload = {"pid": item.pid, "exit_code": item.exit_code}
        elif item.kind == "restart":
            kind = "restart"
            payload = {
                "pid": item.pid,
                "parent_lifetime_id": parents[item.lifetime_number - 1],
                "backoff_ms": item.backoff_ms,
            }
        else:
            kind, payload = "exhausted", {"restarts": item.lifetime_number - 1}
        rows.append((item.monotonic_ns, kind, order, item.lifetime_id, payload))
    rows.append(
        (
            fault.injected_ns,
            "fault",
            -1,
            fault.lifetime_id,
            {
                "pid": fault.pid,
                "signal": fault.signal,
                "acknowledged_responses": fault.acknowledged_responses,
            },
        )
    )
    rows.sort(key=lambda row: (row[0], _TIE_ORDER[row[1]], row[2]))
    if timing.slo_recovered_ns is not None:
        position = next(
            (i for i, row in enumerate(rows) if row[0] > timing.slo_recovered_ns),
            len(rows),
        )
        current = next(
            (row[3] for row in reversed(rows[:position]) if row[1] == "ready"), None
        )
        if current is None:
            raise ServingRecoveryError("SLO recovery precedes every readiness")
        rows.insert(
            position,
            (
                timing.slo_recovered_ns,
                "recovered",
                -1,
                current,
                {
                    "consecutive_successes": config.slo_consecutive_successes,
                    "slo_latency_ms": config.slo_latency_ms,
                },
            ),
        )
    return tuple(
        ServingLifecycleEvent(sequence, run_id, lifetime_id, kind, at, payload)
        for sequence, (at, kind, _, lifetime_id, payload) in enumerate(rows)
    )


def load_lifetime_journals(
    run_dir: Path,
    *,
    observations: tuple[LifecycleObservation, ...],
    capacity: int,
) -> tuple[JournalContents, ...]:
    """Read every spawned lifetime's journal; only killed ones may be torn."""

    journals = []
    for item in observations:
        if item.kind not in ("spawn", "restart"):
            continue
        ended = [
            other
            for other in observations
            if other.lifetime_id == item.lifetime_id
            and other.kind in ("exit_observed", "stopped")
        ]
        became_ready = any(
            other.lifetime_id == item.lifetime_id and other.kind == "ready"
            for other in observations
        )
        path = run_dir / journal_file(item.lifetime_number)
        if not path.exists() and not became_ready:
            journals.append(JournalContents(item.lifetime_id, (), 0))
            continue
        clean = bool(ended) and ended[-1].kind == "stopped" and ended[-1].exit_code == 0
        journals.append(
            read_execution_journal(
                path,
                lifetime_id=item.lifetime_id,
                max_records=capacity,
                allow_torn_tail=not clean,
            )
        )
    return tuple(journals)


def _recovery_boundaries(observations, fault):
    exits = [
        item.monotonic_ns
        for item in observations
        if item.kind == "exit_observed" and item.lifetime_id == fault.lifetime_id
    ]
    replacements = [
        item
        for item in observations
        if item.kind == "ready"
        and lifetime_number(item.lifetime_id) > fault.lifetime_number
    ]
    if not exits or not replacements:
        raise ServingRecoveryError("the faulted server was not observed to recover")
    return exits[0], replacements[0]


def run_serving_recovery_experiment(
    *, source_path: Path, config_path: Path, run_dir: Path
) -> ServingRecoveryResult:
    """Kill a real loopback server under load, restart it and measure recovery."""

    if any(not isinstance(p, Path) for p in (source_path, config_path, run_dir)):
        raise TypeError("experiment paths must be pathlib.Path values")
    if run_dir.exists() or run_dir.is_symlink():
        raise ServingRecoveryError("run_dir must not already exist")
    config = load_serving_recovery_config(config_path)
    source = read_bounded_file(source_path, max_bytes=MAX_SOURCE_BYTES, name="source")
    model = build_serving_model(source)
    run_id = serving_run_id(config, model_sha256=model.model_sha256)
    plan = plan_requests(config, run_id=run_id, model=model)
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + config.max_run_ms * 1_000_000
    run_dir.mkdir(parents=True)
    protected = (source_path, config_path)
    publish_new_file(
        run_dir / SOURCE_FILE, source, max_bytes=MAX_SOURCE_BYTES, protected=protected
    )
    publish_new_file(
        run_dir / CONFIG_FILE,
        canonical_json_bytes(config.to_mapping()),
        max_bytes=16 * 1024,
        protected=protected,
    )

    def command_for(number: int, lifetime_id: str) -> tuple[str, ...]:
        return server_command(
            source_path=run_dir / SOURCE_FILE,
            journal_path=run_dir / journal_file(number),
            lifetime_id=lifetime_id,
            model_sha256=model.model_sha256,
            max_new_tokens=config.max_new_tokens,
            max_records=journal_capacity(config),
        )

    supervisor = ServingSupervisor(
        run_id=run_id,
        config=config,
        model_sha256=model.model_sha256,
        command_for=command_for,
        log_dir=run_dir,
    )
    with supervisor:
        supervisor.start()
        fault = AcknowledgedResponseFault(
            supervisor, after_acknowledged=config.fault_after_acknowledged
        )
        ledger = AttemptLedger(on_acknowledged=fault.observe)
        attempts = run_load(
            plan,
            endpoint=supervisor.endpoint,
            config=config,
            model_sha256=model.model_sha256,
            ledger=ledger,
            deadline_ns=deadline_ns,
        )
        injection = fault.injection
        if injection is None:
            raise ServingRecoveryError("the fault boundary was never reached")
        recovered = supervisor.wait_until(
            lambda: supervisor.failure is not None
            or (
                supervisor.current().ready
                and supervisor.current().number > injection.lifetime_number
            ),
            timeout_ms=max(1, (deadline_ns - time.monotonic_ns()) // 1_000_000),
        )
        if supervisor.failure is not None:
            raise supervisor.failure
        if not recovered:
            raise ServingRecoveryError("no replacement became ready before max_run_ms")
        if supervisor.stop() != 0:
            raise ServingRecoveryError("the replacement did not stop cleanly")
    finished_ns = time.monotonic_ns()
    if finished_ns - started_ns > config.max_run_ms * 1_000_000:
        raise ServingRecoveryError("experiment exceeded max_run_ms")
    observations = supervisor.observations()
    exit_ns, replacement = _recovery_boundaries(observations, injection)
    timing = measure_serving_recovery(
        attempts=attempts,
        fault_injected_ns=injection.injected_ns,
        faulted_lifetime_id=injection.lifetime_id,
        exit_observed_ns=exit_ns,
        replacement_lifetime_id=replacement.lifetime_id,
        replacement_ready_ns=replacement.monotonic_ns,
        slo_latency_ms=config.slo_latency_ms,
        slo_consecutive_successes=config.slo_consecutive_successes,
    )
    journals = load_lifetime_journals(
        run_dir, observations=observations, capacity=journal_capacity(config)
    )
    reconciliation = reconcile_serving_run(
        plan=plan,
        attempts=attempts,
        journals=journals,
        retry_budget=config.retry_budget,
        model_sha256=model.model_sha256,
    )
    events = lifecycle_events(
        run_id=run_id,
        observations=observations,
        fault=injection,
        timing=timing,
        config=config,
    )
    records = chain_serving_trace(events)
    write_attempt_log(run_dir / ATTEMPTS_FILE, attempts, protected=protected)
    write_trace(run_dir / TRACE_FILE, records, protected=protected)
    return ServingRecoveryResult(
        config=config,
        run_id=run_id,
        source_bytes=len(source),
        source_sha256=sha256_hex(source),
        model_sha256=model.model_sha256,
        plan=plan,
        attempts=attempts,
        journals=journals,
        fault=injection,
        events=events,
        trace_sha256=records[-1]["event_sha256"],
        reconciliation=reconciliation,
        timing=timing,
        run_duration_ns=finished_ns - started_ns,
    )
