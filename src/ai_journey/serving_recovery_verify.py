"""Independently verify a serving recovery run from its raw evidence files.

The verifier trusts no stored summary. It rebuilds the model from the copied
source, recomputes the run, lifetime and idempotency identities and the request
plan from the config, checks every evidence file digest, replays the chained
lifecycle trace against the declared restart and SLO controls, re-reads every
lifetime's journal, recomputes each expected response body, then recomputes
reconciliation and every timing metric and requires the sealed report to match
them exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .serving_recovery_client import plan_requests
from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    load_serving_recovery_config,
    read_bounded_file,
    sha256_hex,
)
from .serving_recovery_experiment import (
    ATTEMPTS_FILE,
    CONFIG_FILE,
    SOURCE_FILE,
    TRACE_FILE,
    journal_capacity,
    journal_file,
)
from .serving_recovery_identity import (
    lifetime_number,
    server_lifetime_id,
    serving_run_id,
)
from .serving_recovery_io import (
    MAX_ATTEMPT_LOG_BYTES,
    MAX_TRACE_BYTES,
    load_attempt_log,
    load_trace,
)
from .serving_recovery_journal import JournalContents, read_execution_journal
from .serving_recovery_metrics import PERCENTILE_METHOD, measure_serving_recovery
from .serving_recovery_model import (
    MAX_SOURCE_BYTES,
    build_serving_model,
    completion_body,
)
from .serving_recovery_reconcile import reconcile_serving_run
from .serving_recovery_report import (
    CLOCK,
    EVIDENCE_BOUNDARY,
    REPORT_FILE,
    REPORT_KIND,
    load_serving_report,
    summary_milliseconds,
)

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "experiment",
        "evidence",
        "fault",
        "reconciliation",
        "timing",
        "summary_ms",
        "slo",
        "recovered",
        "clock",
        "percentile_method",
        "evidence_boundary",
        "report_sha256",
    }
)
_EXPERIMENT_FIELDS = frozenset(
    {
        "run_id",
        "config",
        "config_sha256",
        "source_bytes",
        "source_sha256",
        "model_sha256",
        "lifetimes",
        "run_duration_ns",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "source_file",
        "config_file",
        "config_file_sha256",
        "attempts_file",
        "attempts_file_sha256",
        "trace_file",
        "trace_file_sha256",
        "trace_head_sha256",
        "journals",
    }
)
_JOURNAL_FIELDS = frozenset(
    {"lifetime_id", "file", "file_sha256", "records", "torn_tail_bytes"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ServingRecoveryError(message)


def _same(stored: object, recomputed: object) -> bool:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 differ."""

    return canonical_json_bytes(stored) == canonical_json_bytes(recomputed)


def _file_digest(run_dir: Path, name: str, max_bytes: int) -> str:
    return sha256_hex(read_bounded_file(run_dir / name, max_bytes=max_bytes, name=name))


def _journals(run_dir, report, events, capacity):
    lifetimes = [e.lifetime_id for e in events if e.kind in ("spawn", "restart")]
    stored = report["evidence"]["journals"]
    _require(
        type(stored) is list
        and all(type(e) is dict and set(e) == _JOURNAL_FIELDS for e in stored)
        and [entry["lifetime_id"] for entry in stored] == lifetimes,
        "report journals do not match the traced lifetimes",
    )
    journals = []
    for lifetime_id, entry in zip(lifetimes, stored):
        name = journal_file(lifetime_number(lifetime_id))
        ready = any(e.kind == "ready" and e.lifetime_id == lifetime_id for e in events)
        clean = any(
            e.kind == "finished"
            and e.lifetime_id == lifetime_id
            and e.payload["exit_code"] == 0
            for e in events
        )
        if not (run_dir / name).exists():
            _require(not ready, "a ready lifetime is missing its journal")
            missing = {
                "lifetime_id": lifetime_id,
                "file": None,
                "file_sha256": None,
                "records": 0,
                "torn_tail_bytes": 0,
            }
            _require(_same(entry, missing), "report misstates a missing journal")
            journals.append(JournalContents(lifetime_id, (), 0))
            continue
        _require(entry["file"] == name, "report journal file name differs")
        _require(
            entry["file_sha256"] == _file_digest(run_dir, name, MAX_ATTEMPT_LOG_BYTES),
            "journal file digest does not match the report",
        )
        journal = read_execution_journal(
            run_dir / name,
            lifetime_id=lifetime_id,
            max_records=capacity,
            allow_torn_tail=not clean,
        )
        _require(
            _same(
                (entry["records"], entry["torn_tail_bytes"]),
                (len(journal.records), journal.torn_tail_bytes),
            ),
            "report journal counts do not match the journal",
        )
        journals.append(journal)
    return tuple(journals)


def _check_attempt_timeline(attempts, events) -> None:
    """Bind every attempt to a published port inside the traced lifecycle."""

    ports = {e.lifetime_id: e.payload["port"] for e in events if e.kind == "ready"}
    _require(bool(attempts) and bool(ports), "run has no attempts or readiness")
    for attempt in attempts:
        _require(
            attempt.port in ports.values(),
            "an attempt used a port no lifetime published",
        )
        _require(
            attempt.outcome != "ok" or ports.get(attempt.lifetime_id) == attempt.port,
            "an acknowledgement names a lifetime that did not own its port",
        )
    first_ready = next(e.monotonic_ns for e in events if e.kind == "ready")
    _require(
        first_ready <= min(a.started_ns for a in attempts)
        and max(a.finished_ns for a in attempts) <= events[-1].monotonic_ns,
        "attempts fall outside the traced server lifecycle",
    )


def _fault_boundaries(events, attempts, config):
    faults = [e for e in events if e.kind == "fault"]
    _require(len(faults) == 1, "trace must contain exactly one fault")
    fault = faults[0]
    _require(
        fault.payload["acknowledged_responses"] == config.fault_after_acknowledged,
        "fault boundary differs from the config",
    )
    acknowledged = [a for a in attempts if a.outcome == "ok"]
    _require(
        len(acknowledged) >= config.fault_after_acknowledged,
        "attempt log has fewer acknowledgements than the fault boundary",
    )
    trigger = acknowledged[config.fault_after_acknowledged - 1]
    _require(
        trigger.lifetime_id == fault.lifetime_id
        and trigger.finished_ns <= fault.monotonic_ns,
        "the K-th acknowledgement did not precede the fault on its lifetime",
    )
    after = [e for e in events if e.sequence > fault.sequence]
    exits = [
        e
        for e in after
        if e.kind == "exit_observed" and e.lifetime_id == fault.lifetime_id
    ]
    readies = [
        e
        for e in after
        if e.kind == "ready"
        and lifetime_number(e.lifetime_id) > lifetime_number(fault.lifetime_id)
    ]
    _require(bool(exits) and bool(readies), "trace shows no observed recovery")
    return fault, exits[0], readies[0]


def _check_traced_controls(events, config) -> None:
    """Bind traced restart backoffs and SLO payloads to the declared controls."""

    restart = 0
    for event in events:
        if event.kind == "restart":
            # A restart whose spawn failed is untraced but still spends its delay.
            restart = next(
                (
                    number
                    for number in range(restart + 1, config.restart_limit + 1)
                    if config.restart_delay_ms(number) == event.payload["backoff_ms"]
                ),
                0,
            )
            _require(restart > 0, "a restart breaks the declared backoff schedule")
        elif event.kind == "recovered":
            _require(
                event.payload
                == {
                    "consecutive_successes": config.slo_consecutive_successes,
                    "slo_latency_ms": config.slo_latency_ms,
                },
                "the traced SLO recovery names other controls",
            )


def verify_serving_recovery_run(run_dir: Path) -> dict[str, Any]:
    """Recompute every identity, digest and derived metric of one run.

    The wall-clock run duration has no independent raw record, so it is bounded
    by the traced lifecycle span and ``max_run_ms`` rather than recomputed.
    """

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    _require(run_dir.is_dir() and not run_dir.is_symlink(), "run_dir is invalid")
    report = load_serving_report(run_dir / REPORT_FILE)
    _require(set(report) == _REPORT_FIELDS, "report fields do not match schema")
    _require(
        type(report["schema_version"]) is int
        and report["schema_version"] == 1
        and report["kind"] == REPORT_KIND,
        "report schema is unsupported",
    )
    _require(report["evidence_boundary"] == EVIDENCE_BOUNDARY, "boundary changed")
    _require(report["clock"] == CLOCK, "report clock statement changed")
    _require(
        report["percentile_method"] == PERCENTILE_METHOD, "percentile method changed"
    )
    experiment, evidence = report["experiment"], report["evidence"]
    _require(
        type(experiment) is dict
        and set(experiment) == _EXPERIMENT_FIELDS
        and type(evidence) is dict
        and set(evidence) == _EVIDENCE_FIELDS,
        "report sections do not match schema",
    )
    _require(
        (
            evidence["source_file"],
            evidence["config_file"],
            evidence["attempts_file"],
            evidence["trace_file"],
        )
        == (SOURCE_FILE, CONFIG_FILE, ATTEMPTS_FILE, TRACE_FILE),
        "report names unexpected evidence files",
    )

    config = load_serving_recovery_config(run_dir / CONFIG_FILE)
    _require(_same(experiment["config"], config.to_mapping()), "config differs")
    _require(experiment["config_sha256"] == config.fingerprint(), "config digest")
    _require(
        evidence["config_file_sha256"] == _file_digest(run_dir, CONFIG_FILE, 16 * 1024),
        "config file digest does not match the report",
    )
    source = read_bounded_file(
        run_dir / SOURCE_FILE, max_bytes=MAX_SOURCE_BYTES, name="source"
    )
    _require(
        _same(
            (experiment["source_bytes"], experiment["source_sha256"]),
            (len(source), sha256_hex(source)),
        ),
        "source copy does not match the report",
    )
    model = build_serving_model(source)
    _require(experiment["model_sha256"] == model.model_sha256, "model differs")
    run_id = serving_run_id(config, model_sha256=model.model_sha256)
    _require(experiment["run_id"] == run_id, "run identity does not recompute")
    plan = plan_requests(config, run_id=run_id, model=model)

    attempts = load_attempt_log(run_dir / ATTEMPTS_FILE)
    _require(
        evidence["attempts_file_sha256"]
        == _file_digest(run_dir, ATTEMPTS_FILE, MAX_ATTEMPT_LOG_BYTES),
        "attempt log digest does not match the report",
    )
    records, events = load_trace(run_dir / TRACE_FILE)
    _require(
        evidence["trace_file_sha256"]
        == _file_digest(run_dir, TRACE_FILE, MAX_TRACE_BYTES)
        and evidence["trace_head_sha256"] == records[-1]["event_sha256"],
        "trace digests do not match the report",
    )
    _require(events[0].run_id == run_id, "trace belongs to another run")
    _require(events[-1].payload["exit_code"] == 0, "the final lifetime did not stop")
    lifetimes = [e.lifetime_id for e in events if e.kind in ("spawn", "restart")]
    _require(
        lifetimes == [server_lifetime_id(run_id, n + 1) for n in range(len(lifetimes))]
        and experiment["lifetimes"] == lifetimes,
        "lifetime identities do not recompute",
    )
    journals = _journals(run_dir, report, events, journal_capacity(config))
    _check_traced_controls(events, config)
    _check_attempt_timeline(attempts, events)

    expected = {
        p.idempotency_key: sha256_hex(completion_body(model, p.request)) for p in plan
    }
    for attempt in attempts:
        _require(
            attempt.outcome != "ok"
            or attempt.response_sha256 == expected.get(attempt.idempotency_key),
            "an acknowledged response differs from the recomputed output",
        )
    for journal in journals:
        for record in journal.records:
            _require(
                record.response_sha256 == expected.get(record.idempotency_key),
                "an executed response differs from the recomputed output",
            )

    reconciliation = reconcile_serving_run(
        plan=plan,
        attempts=attempts,
        journals=journals,
        retry_budget=config.retry_budget,
        model_sha256=model.model_sha256,
    )
    _require(
        _same(report["reconciliation"], reconciliation.to_mapping()),
        "reconciliation does not match the raw evidence",
    )
    fault, exit_event, ready_event = _fault_boundaries(events, attempts, config)
    _require(
        _same(
            report["fault"],
            {
                "mechanism": "process_kill",
                "signal": fault.payload["signal"],
                "lifetime_id": fault.lifetime_id,
                "acknowledged_responses": fault.payload["acknowledged_responses"],
                "injected_ns": fault.monotonic_ns,
            },
        ),
        "fault section does not match the trace",
    )
    timing = measure_serving_recovery(
        attempts=attempts,
        fault_injected_ns=fault.monotonic_ns,
        faulted_lifetime_id=fault.lifetime_id,
        exit_observed_ns=exit_event.monotonic_ns,
        replacement_lifetime_id=ready_event.lifetime_id,
        replacement_ready_ns=ready_event.monotonic_ns,
        slo_latency_ms=config.slo_latency_ms,
        slo_consecutive_successes=config.slo_consecutive_successes,
    ).to_mapping()
    _require(_same(report["timing"], timing), "timing does not match the raw evidence")
    _require(
        _same(report["summary_ms"], summary_milliseconds(timing)),
        "millisecond summary does not match the timing",
    )
    recovered_events = [e for e in events if e.kind == "recovered"]
    slo_recovered = timing["slo_recovered_ns"] is not None
    _require(
        [e.monotonic_ns for e in recovered_events]
        == ([timing["slo_recovered_ns"]] if slo_recovered else []),
        "trace recovery does not match the measured SLO recovery",
    )
    _require(
        _same(
            report["slo"],
            {
                "latency_ms": config.slo_latency_ms,
                "consecutive_successes": config.slo_consecutive_successes,
                "recovered": slo_recovered,
            },
        ),
        "SLO section does not match the measurement",
    )
    recovered = slo_recovered and reconciliation.totals["failed"] == 0
    _require(report["recovered"] is recovered, "recovered flag does not recompute")
    duration = experiment["run_duration_ns"]
    _require(
        type(duration) is int
        and events[-1].monotonic_ns - events[0].monotonic_ns
        <= duration
        <= config.max_run_ms * 1_000_000,
        "run duration is inconsistent with the trace or max_run_ms",
    )
    return {
        "verified": True,
        "run_id": run_id,
        "report_sha256": report["report_sha256"],
        "recovered": recovered,
        "requests": reconciliation.totals["requests"],
        "duplicate_executions": reconciliation.totals["duplicate_executions"],
        "summary_ms": report["summary_ms"],
    }
