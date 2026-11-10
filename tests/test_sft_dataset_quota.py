import pytest

from ai_journey.sft_dataset_quota import category_quotas
from ai_journey.sft_dataset_record import SFTDatasetError


def test_quotas_use_largest_remainders_and_reconcile_exactly():
    quotas = category_quotas({"b": 20, "a": 70, "c": 10}, 17)
    assert quotas == {"a": 12, "b": 3, "c": 2}
    assert sum(quotas.values()) == 17


def test_quotas_preserve_rare_categories_and_respect_capacity():
    quotas = category_quotas({"large": 100, "rare": 1, "small": 2}, 8)
    assert quotas["rare"] == 1
    assert quotas["small"] == 1
    assert quotas["large"] == 6


def test_quotas_are_independent_of_mapping_insertion_order():
    forward = category_quotas({"a": 11, "b": 7, "c": 3}, 10)
    reverse = category_quotas({"c": 3, "b": 7, "a": 11}, 10)
    assert forward == reverse


def test_target_equal_to_categories_assigns_one_each():
    assert category_quotas({"a": 100, "b": 2, "c": 1}, 3) == {
        "a": 1,
        "b": 1,
        "c": 1,
    }


@pytest.mark.parametrize(
    "counts,target",
    [
        ({}, 1),
        ({"a": 0}, 1),
        ({"a": True}, 1),
        ({"a": 1, "b": 1}, 1),
        ({"a": 1}, 2),
        ({"a": 2}, True),
    ],
)
def test_invalid_quota_inputs_are_rejected(counts, target):
    with pytest.raises(SFTDatasetError):
        category_quotas(counts, target)
