"""Reconcile client attempts with every server lifetime's execution journal.

Each logical request ends as a first-try success, a success after retry, or a
failure after its whole retry budget. Committed journal records are matched to
the exact client attempt (idempotency key and attempt number) that caused them,
across all lifetimes. An execution whose attempt did not end ``ok`` lost its
response, typically to the injected kill; a later retry then executes the same
key again, an at-least-once duplicate that is counted explicitly. Every
acknowledged and every executed response of one key must be byte-identical.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .serving_recovery_client import ATTEMPT_OUTCOMES, AttemptRecord, PlannedRequest
from .serving_recovery_config import ServingRecoveryError, require_digest, require_int
from .serving_recovery_journal import JournalContents

REQUEST_OUTCOMES = ("first_try_success", "success_after_retry", "failed_after_budget")


@dataclass(frozen=True)
class RequestReconciliation:
    """Client and server accounting for one logical request."""

    request_index: int
    idempotency_key: str
    outcome: str
    attempts: int
    executions: int
    lost_response_executions: int
    acknowledged_lifetime_id: str | None
    response_sha256: str | None


@dataclass(frozen=True)
class ServingReconciliation:
    """Per-request outcomes plus the run's accounting identities."""

    requests: tuple[RequestReconciliation, ...]
    totals: dict[str, int]
    attempt_outcomes: dict[str, int]
    executions_by_lifetime: dict[str, int]
    torn_tail_bytes_by_lifetime: dict[str, int]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requests": [asdict(request) for request in self.requests],
            "totals": dict(self.totals),
            "attempt_outcomes": dict(self.attempt_outcomes),
            "executions_by_lifetime": dict(self.executions_by_lifetime),
            "torn_tail_bytes_by_lifetime": dict(self.torn_tail_bytes_by_lifetime),
        }


def _request_attempts(
    plan: Sequence[PlannedRequest], attempts: Sequence[AttemptRecord]
) -> dict[str, list[AttemptRecord]]:
    if [planned.index for planned in plan] != list(range(len(plan))):
        raise ServingRecoveryError("plan indices must be contiguous from zero")
    keyed: dict[str, list[AttemptRecord]] = {p.idempotency_key: [] for p in plan}
    for attempt in attempts:
        if not isinstance(attempt, AttemptRecord):
            raise TypeError("attempts must be AttemptRecord values")
        if attempt.idempotency_key not in keyed:
            raise ServingRecoveryError("attempt carries a key outside the plan")
        keyed[attempt.idempotency_key].append(attempt)
    return keyed


def _executions(
    plan: Sequence[PlannedRequest],
    journals: Sequence[JournalContents],
    attempted: dict[tuple[str, int], AttemptRecord],
    model_sha256: str,
):
    digests = {p.idempotency_key: p.request.request_sha256() for p in plan}
    if len({journal.lifetime_id for journal in journals}) != len(journals):
        raise ServingRecoveryError("each lifetime must contribute one journal")
    executed = {}
    for journal in journals:
        if not isinstance(journal, JournalContents):
            raise TypeError("journals must be JournalContents values")
        for record in journal.records:
            key = (record.idempotency_key, record.attempt)
            if record.idempotency_key not in digests:
                raise ServingRecoveryError("journal executes a key outside the plan")
            if key in executed:
                raise ServingRecoveryError("one client attempt executed twice")
            if key not in attempted:
                raise ServingRecoveryError("journal executes an attempt never sent")
            if record.request_sha256 != digests[record.idempotency_key]:
                raise ServingRecoveryError("journal request differs from the plan")
            if record.model_sha256 != model_sha256:
                raise ServingRecoveryError("journal served an unexpected model")
            executed[key] = record
    return executed


def reconcile_serving_run(
    *,
    plan: Sequence[PlannedRequest],
    attempts: Sequence[AttemptRecord],
    journals: Sequence[JournalContents],
    retry_budget: int,
    model_sha256: str,
) -> ServingReconciliation:
    """Recompute request outcomes, execution counts and duplicate accounting."""

    require_int(retry_budget, "retry_budget", 1, 64)
    require_digest(model_sha256, "model_sha256")
    keyed = _request_attempts(plan, attempts)
    attempted = {(a.idempotency_key, a.attempt): a for a in attempts}
    if len(attempted) != len(attempts):
        raise ServingRecoveryError("an attempt number repeats within a request")
    executed = _executions(plan, journals, attempted, model_sha256)
    per_key = defaultdict(list)
    for (key, _), record in executed.items():
        per_key[key].append(record)
    rows = []
    for planned in plan:
        history = keyed[planned.idempotency_key]
        if not history:
            raise ServingRecoveryError("a planned request was never attempted")
        if [a.attempt for a in history] != list(range(1, len(history) + 1)):
            raise ServingRecoveryError("request attempts are not numbered 1..n")
        oks = [a for a in history if a.outcome == "ok"]
        if oks and oks != [history[-1]]:
            raise ServingRecoveryError("a request continued after its success")
        if oks:
            outcome = (
                "first_try_success" if len(history) == 1 else "success_after_retry"
            )
            match = executed.get((planned.idempotency_key, oks[0].attempt))
            if match is None or match.lifetime_id != oks[0].lifetime_id:
                raise ServingRecoveryError("an acknowledgement has no execution")
        elif len(history) == retry_budget + 1:
            outcome = "failed_after_budget"
        else:
            raise ServingRecoveryError("a request stopped before its budget ended")
        records = per_key[planned.idempotency_key]
        digests = {r.response_sha256 for r in records} | {
            a.response_sha256 for a in oks
        }
        if len(digests) > 1:
            raise ServingRecoveryError("retried responses are not byte-identical")
        rows.append(
            RequestReconciliation(
                request_index=planned.index,
                idempotency_key=planned.idempotency_key,
                outcome=outcome,
                attempts=len(history),
                executions=len(records),
                lost_response_executions=sum(
                    attempted[(r.idempotency_key, r.attempt)].outcome != "ok"
                    for r in records
                ),
                acknowledged_lifetime_id=oks[0].lifetime_id if oks else None,
                response_sha256=next(iter(digests), None),
            )
        )
    outcomes = Counter(row.outcome for row in rows)
    totals = {
        "requests": len(rows),
        "succeeded": outcomes["first_try_success"] + outcomes["success_after_retry"],
        "failed": outcomes["failed_after_budget"],
        **{name: outcomes[name] for name in REQUEST_OUTCOMES},
        "attempts": len(attempts),
        "retried_attempts": len(attempts) - len(rows),
        "executions": len(executed),
        "duplicate_executions": sum(max(0, row.executions - 1) for row in rows),
        "requests_executed_more_than_once": sum(row.executions > 1 for row in rows),
        "lost_response_executions": sum(row.lost_response_executions for row in rows),
        "unexecuted_failed_attempts": sum(
            a.outcome != "ok" and (a.idempotency_key, a.attempt) not in executed
            for a in attempts
        ),
        "torn_journal_tails": sum(j.torn_tail_bytes > 0 for j in journals),
    }
    if totals["requests"] != totals["succeeded"] + totals["failed"]:
        raise ServingRecoveryError("requests do not equal succeeded plus failed")
    if totals["attempts"] != sum(row.attempts for row in rows):
        raise ServingRecoveryError("attempts do not equal per-request attempts")
    if totals["executions"] != totals["succeeded"] + totals["lost_response_executions"]:
        raise ServingRecoveryError("executions do not equal acks plus lost responses")
    attempt_outcomes = Counter(a.outcome for a in attempts)
    return ServingReconciliation(
        requests=tuple(rows),
        totals=totals,
        attempt_outcomes={name: attempt_outcomes[name] for name in ATTEMPT_OUTCOMES},
        executions_by_lifetime={j.lifetime_id: len(j.records) for j in journals},
        torn_tail_bytes_by_lifetime={
            j.lifetime_id: j.torn_tail_bytes for j in journals
        },
    )
