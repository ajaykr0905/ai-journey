from ai_journey.verifiable_reward_attempt import RewardAttempt
from ai_journey.verifiable_reward_metrics import summarize_attempts


def test_metrics_count_final_candidates_without_double_counting_retries():
    attempts = (
        RewardAttempt("t", 0, 1, "transient_failure", 0, "injected"),
        RewardAttempt("t", 0, 2, "correct", 1, "correct"),
        RewardAttempt("t", 1, 1, "incorrect", 0.2, "incorrect"),
    )
    metrics = summarize_attempts(attempts)
    assert metrics == {
        "attempt_count": 3,
        "candidate_count": 2,
        "retry_count": 1,
        "correct_count": 1,
        "format_error_count": 0,
        "terminal_failure_count": 0,
        "mean_reward": 0.6,
    }


def test_empty_metrics_do_not_divide_by_zero():
    assert summarize_attempts(())["mean_reward"] == 0.0
