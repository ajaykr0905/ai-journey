import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_task import VerifiableTask


def payload():
    return {
        "task_id": "checkpoint-loss-01",
        "prompt": "Return the number of lost committed steps.",
        "verifier": "numeric",
        "expected": 0,
        "options": {"absolute_tolerance": "0"},
    }


def test_task_round_trips_strict_public_record():
    task = VerifiableTask.from_mapping(payload())

    assert task.to_mapping() == payload()


@pytest.mark.parametrize("task_id", ["", "UPPER", "space id", "a" * 65])
def test_task_rejects_noncanonical_identity(task_id):
    value = payload()
    value["task_id"] = task_id
    with pytest.raises(VerifiableRewardError, match="task_id"):
        VerifiableTask.from_mapping(value)


def test_task_rejects_unknown_fields_and_nonfinite_expected_value():
    value = payload()
    value["extra"] = "hidden"
    with pytest.raises(VerifiableRewardError, match="fields"):
        VerifiableTask.from_mapping(value)

    value = payload()
    value["expected"] = float("nan")
    with pytest.raises(VerifiableRewardError, match="finite JSON"):
        VerifiableTask.from_mapping(value)
