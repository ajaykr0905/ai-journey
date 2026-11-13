from dataclasses import replace
from hashlib import sha256

import pytest
from test_serving_recovery_client import MODEL, PLAN, RUN

from ai_journey.serving_recovery_client import AttemptRecord
from ai_journey.serving_recovery_identity import server_lifetime_id
from ai_journey.serving_recovery_journal import ExecutionRecord, JournalContents
from ai_journey.serving_recovery_model import completion_body
from ai_journey.serving_recovery_reconcile import reconcile_serving_run

FIRST = server_lifetime_id(RUN, 1)
SECOND = server_lifetime_id(RUN, 2)
BUDGET = 14


def digest(index):
    return sha256(completion_body(MODEL, PLAN[index].request)).hexdigest()


def attempt(index, number, outcome, lifetime=None, start=0):
    ok = outcome == "ok"
    return AttemptRecord(
        request_index=index,
        idempotency_key=PLAN[index].idempotency_key,
        attempt=number,
        port=8080,
        started_ns=start,
        finished_ns=start + 10,
        outcome=outcome,
        error=None if ok else "scripted",
        http_status=200 if ok else None,
        lifetime_id=lifetime if ok else None,
        response_sha256=digest(index) if ok else None,
    )


def execution(lifetime, sequence, index, number):
    return ExecutionRecord(
        lifetime_id=lifetime,
        sequence=sequence,
        idempotency_key=PLAN[index].idempotency_key,
        attempt=number,
        request_sha256=PLAN[index].request.request_sha256(),
        response_sha256=digest(index),
        model_sha256=MODEL.model_sha256,
    )


def scenario():
    """Request 5 lost its response to the kill; 7 never reached a server."""

    attempts, first, second = [], [], []
    for index in range(len(PLAN)):
        if index == 5:
            attempts += [
                attempt(5, 1, "connection_reset"),
                attempt(5, 2, "connection_refused"),
                attempt(5, 3, "ok", SECOND),
            ]
            first.append((5, 1))
            second.append((5, 3))
        elif index == 6:
            attempts += [
                attempt(6, 1, "connection_refused"),
                attempt(6, 2, "ok", SECOND),
            ]
            second.append((6, 2))
        elif index == 7:
            attempts += [
                attempt(7, number, "connection_refused")
                for number in range(1, BUDGET + 2)
            ]
        else:
            lifetime = FIRST if index < 5 else SECOND
            attempts.append(attempt(index, 1, "ok", lifetime))
            (first if index < 5 else second).append((index, 1))
    journals = [
        JournalContents(
            lifetime,
            tuple(
                execution(lifetime, sequence, index, number)
                for sequence, (index, number) in enumerate(executed)
            ),
            torn,
        )
        for lifetime, executed, torn in ((FIRST, first, 7), (SECOND, second, 0))
    ]
    return attempts, journals


def reconcile(attempts, journals, plan=PLAN):
    return reconcile_serving_run(
        plan=plan,
        attempts=attempts,
        journals=journals,
        retry_budget=BUDGET,
        model_sha256=MODEL.model_sha256,
    )


def test_reconciliation_counts_outcomes_duplicates_and_lost_responses():
    result = reconcile(*scenario())
    rows = {row.request_index: row for row in result.requests}

    assert result.totals == {
        "requests": 24,
        "succeeded": 23,
        "failed": 1,
        "first_try_success": 21,
        "success_after_retry": 2,
        "failed_after_budget": 1,
        "attempts": 21 + 3 + 2 + 15,
        "retried_attempts": 41 - 24,
        "executions": 24,
        "duplicate_executions": 1,
        "requests_executed_more_than_once": 1,
        "lost_response_executions": 1,
        "unexecuted_failed_attempts": 17,
        "torn_journal_tails": 1,
    }
    assert (rows[5].outcome, rows[5].executions, rows[5].lost_response_executions) == (
        "success_after_retry",
        2,
        1,
    )
    assert rows[5].acknowledged_lifetime_id == SECOND
    assert rows[5].response_sha256 == digest(5)
    assert (rows[7].outcome, rows[7].attempts, rows[7].executions) == (
        "failed_after_budget",
        15,
        0,
    )
    assert rows[7].response_sha256 is None
    assert result.attempt_outcomes == {
        "ok": 23,
        "connection_refused": 1 + 1 + 15,
        "connection_reset": 1,
        "timeout": 0,
        "http_error": 0,
        "invalid_response": 0,
    }
    assert result.executions_by_lifetime == {FIRST: 6, SECOND: 18}
    assert result.torn_tail_bytes_by_lifetime == {FIRST: 7, SECOND: 0}
    assert result.to_mapping()["totals"] == result.totals


def changed_journal(journals, position, **changes):
    first, second = journals
    records = list(first.records)
    records[position] = replace(records[position], **changes)
    return [replace(first, records=tuple(records)), second]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda a, j: (a, changed_journal(j, -1, response_sha256="0" * 64)),
            "byte-identical",
        ),
        (lambda a, j: (a, changed_journal(j, -1, attempt=9)), "never sent"),
        (lambda a, j: (a, changed_journal(j, 0, request_sha256="0" * 64)), "plan"),
        (lambda a, j: (a, changed_journal(j, 0, model_sha256="0" * 64)), "model"),
        (lambda a, j: (a, [j[0], j[0]]), "one journal"),
        (lambda a, j: (a, [j[0], replace(j[1], records=())]), "no execution"),
        (lambda a, j: ([x for x in a if x.attempt != 15], j), "budget"),
        (lambda a, j: ([x for x in a if x.request_index != 7], j), "never attempted"),
        (lambda a, j: (a + [attempt(0, 2, "timeout")], j), "after its success"),
        (lambda a, j: (a + [attempt(0, 1, "timeout")], j), "repeats"),
        (lambda a, j: ([replace(a[0], lifetime_id=SECOND)] + a[1:], j), "execution"),
    ],
)
def test_reconciliation_rejects_unaccountable_evidence(mutate, message):
    attempts, journals = mutate(*scenario())
    with pytest.raises(ValueError, match=message):
        reconcile(attempts, journals)


def test_reconciliation_requires_a_complete_ordered_plan():
    attempts, journals = scenario()
    with pytest.raises(ValueError, match="contiguous"):
        reconcile(attempts, journals, plan=PLAN[1:])
    with pytest.raises(ValueError, match="outside the plan"):
        reconcile(attempts, journals, plan=PLAN[:5])
