import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_identity import sha256_json, task_set_identity
from ai_journey.verifiable_reward_task import VerifiableTask


def task(task_id="a"):
    return VerifiableTask.from_mapping(
        {
            "task_id": task_id,
            "prompt": "p",
            "verifier": "exact",
            "expected": "x",
            "options": {},
        }
    )


def test_json_identity_is_key_order_independent_but_list_order_sensitive():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    assert sha256_json([1, 2]) != sha256_json([2, 1])


def test_task_set_identity_binds_order_and_rejects_duplicates():
    assert task_set_identity([task("a"), task("b")]) != task_set_identity(
        [task("b"), task("a")]
    )
    with pytest.raises(VerifiableRewardError, match="unique"):
        task_set_identity([task(), task()])
