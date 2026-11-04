import pytest

from ai_journey.inference_kv import InferenceCapacityError
from ai_journey.inference_paging import PagedAllocation, plan_paged_allocation


def test_paged_allocation_exposes_per_sequence_internal_fragmentation():
    allocation = plan_paged_allocation(
        (1, 16, 17), block_tokens=16, bytes_per_token=128 * 1024
    )

    assert allocation == PagedAllocation(
        requested_tokens=34,
        blocks=4,
        block_tokens=16,
        allocated_tokens=64,
        internal_waste_tokens=30,
        allocated_bytes=8 * 1024**2,
    )


def test_full_pages_have_no_internal_fragmentation():
    allocation = plan_paged_allocation((16, 32), block_tokens=16, bytes_per_token=1024)

    assert allocation.internal_waste_tokens == 0
    assert allocation.allocated_tokens == allocation.requested_tokens


@pytest.mark.parametrize("lengths", [(), [1, 2]])
def test_page_planner_requires_an_explicit_nonempty_tuple(lengths):
    with pytest.raises(InferenceCapacityError, match="nonempty tuple"):
        plan_paged_allocation(lengths, block_tokens=16, bytes_per_token=1024)
