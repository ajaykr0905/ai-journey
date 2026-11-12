from ai_journey.verifiable_reward_registry import verify_response
from ai_journey.verifiable_reward_task import VerifiableTask


def task(kind, expected, options=None):
    return VerifiableTask.from_mapping(
        {
            "task_id": kind,
            "prompt": "p",
            "verifier": kind,
            "expected": expected,
            "options": options or {},
        }
    )


def test_registry_dispatches_each_verifier_kind():
    assert verify_response(task("exact", "ok"), "FINAL: ok", max_bytes=100).correct
    assert verify_response(task("numeric", 3), "FINAL: 3.0", max_bytes=100).correct
    assert verify_response(
        task("json", {"ok": True}), 'FINAL: {"ok":true}', max_bytes=100
    ).correct


def test_registry_separates_bad_format_from_wrong_answer():
    malformed = verify_response(task("exact", "ok"), "ok", max_bytes=100)
    wrong = verify_response(task("exact", "ok"), "FINAL: no", max_bytes=100)
    assert (malformed.format_valid, malformed.correct) == (False, False)
    assert (wrong.format_valid, wrong.correct) == (True, False)
