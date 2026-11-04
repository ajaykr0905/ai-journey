from fractions import Fraction

import pytest

from ai_journey.inference_kv import InferenceCapacityError
from ai_journey.inference_paging import (
    PagedAllocation,
    admit_paged_allocation,
    plan_paged_allocation,
    plan_shared_prefix,
)


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
    assert allocation.internal_waste_fraction == Fraction(15, 32)


def test_full_pages_have_no_internal_fragmentation():
    allocation = plan_paged_allocation((16, 32), block_tokens=16, bytes_per_token=1024)

    assert allocation.internal_waste_tokens == 0
    assert allocation.allocated_tokens == allocation.requested_tokens


@pytest.mark.parametrize("lengths", [(), [1, 2]])
def test_page_planner_requires_an_explicit_nonempty_tuple(lengths):
    with pytest.raises(InferenceCapacityError, match="nonempty tuple"):
        plan_paged_allocation(lengths, block_tokens=16, bytes_per_token=1024)


def test_page_admission_charges_whole_blocks_instead_of_requested_tokens():
    allocation = plan_paged_allocation(
        (1, 16, 17), block_tokens=16, bytes_per_token=128 * 1024
    )

    assert admit_paged_allocation(allocation, capacity_bytes=8 * 1024**2).admitted
    assert not admit_paged_allocation(
        allocation, capacity_bytes=8 * 1024**2 - 1
    ).admitted


def test_shared_prefix_is_charged_once_across_a_batch():
    allocation = plan_shared_prefix(
        (32_768,) * 16,
        shared_prefix_tokens=8_192,
        bytes_per_token=128 * 1024,
    )

    assert allocation.logical_tokens == 16 * 32_768
    assert allocation.saved_tokens == 15 * 8_192
    assert (
        allocation.physical_tokens
        == allocation.logical_tokens - allocation.saved_tokens
    )
    assert allocation.physical_bytes == allocation.physical_tokens * 128 * 1024


def test_shared_prefix_rejects_a_shorter_sequence():
    with pytest.raises(InferenceCapacityError, match="cannot exceed"):
        plan_shared_prefix((8, 16), shared_prefix_tokens=9, bytes_per_token=1024)
