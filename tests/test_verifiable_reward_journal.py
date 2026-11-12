from ai_journey.verifiable_reward_attempt import RewardAttempt
from ai_journey.verifiable_reward_journal import (
    build_attempt_journal,
    verify_attempt_journal,
)


def test_journal_binds_order_content_and_previous_event():
    events = build_attempt_journal(
        (RewardAttempt("t", 0, 1, "correct", 1.0, "correct"),)
    )
    assert verify_attempt_journal(events)
    events[0]["attempt"]["reward"] = 0.0
    assert not verify_attempt_journal(events)


def test_empty_journal_is_valid_genesis_state():
    assert verify_attempt_journal([])
