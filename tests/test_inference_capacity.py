import pytest

from ai_journey.inference_capacity import (
    DeviceMemoryBudget,
    maximum_resident_sequences,
)
from ai_journey.inference_kv import InferenceCapacityError, KVCacheLayout

GIB = 1024**3


def test_device_budget_limits_full_context_mha_and_gqa_batch_capacity():
    budget = DeviceMemoryBudget(
        80 * GIB, model_bytes=14 * GIB, runtime_reserve_bytes=6 * GIB
    )

    assert (
        maximum_resident_sequences(budget, KVCacheLayout(32, 32, 128, 32_768, 1)) == 3
    )
    assert (
        maximum_resident_sequences(budget, KVCacheLayout(32, 8, 128, 32_768, 1)) == 15
    )


def test_device_budget_rejects_reserves_that_consume_the_device():
    with pytest.raises(InferenceCapacityError, match="positive KV"):
        DeviceMemoryBudget(80 * GIB, 70 * GIB, 10 * GIB)
