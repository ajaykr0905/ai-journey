import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_accumulator import (
    EMPTY_ACCUMULATOR_SHA256,
    accumulate_work_item,
    rebuild_accumulator,
)
from ai_journey.recovery_workload import build_recovery_workload


def workload():
    return build_recovery_workload(b"public recovery fixture", valid_config())


def test_accumulator_is_ordered_and_reconstructs_every_prefix():
    data = workload()
    digest = EMPTY_ACCUMULATOR_SHA256
    assert rebuild_accumulator(data, 0) == digest
    for completed, item in enumerate(data.items, start=1):
        digest = accumulate_work_item(digest, item)
        assert rebuild_accumulator(data, completed) == digest


def test_accumulator_changes_when_item_order_changes():
    data = workload()
    first = accumulate_work_item(EMPTY_ACCUMULATOR_SHA256, data.items[0])
    first = accumulate_work_item(first, data.items[1])
    second = accumulate_work_item(EMPTY_ACCUMULATOR_SHA256, data.items[1])
    second = accumulate_work_item(second, data.items[0])
    assert first != second


@pytest.mark.parametrize("completed", [True, -1, 21])
def test_rebuild_rejects_invalid_prefix_bounds(completed):
    with pytest.raises((TypeError, ValueError), match="integer|outside"):
        rebuild_accumulator(workload(), completed)


def test_accumulator_rejects_invalid_parent_digest_and_item_type():
    with pytest.raises(ValueError, match="SHA-256"):
        accumulate_work_item("bad", workload().items[0])
    with pytest.raises(TypeError, match="RecoveryWorkItem"):
        accumulate_work_item(EMPTY_ACCUMULATOR_SHA256, object())
