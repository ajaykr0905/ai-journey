import pytest

from ai_journey.memory_access_banks import (
    BANK_EXCLUSION,
    analyze_banks,
    summarize_banks,
)
from ai_journey.memory_access_pattern import (
    MemoryAccessError,
    WarpAccess,
    broadcast_access,
    contiguous_access,
    lane_mask,
    permuted_access,
    strided_access,
)


def words(*indices: int, width: int = 4) -> WarpAccess:
    return WarpAccess(width, tuple(index * 4 for index in indices))


def test_contiguous_float32_words_spread_over_every_bank():
    analysis = analyze_banks(contiguous_access())

    assert analysis.conflict_free
    assert analysis.conflict_degree == 1
    assert analysis.wavefronts == 1
    assert analysis.distinct_words == 32
    assert analysis.banks_used == 32
    assert analysis.multicast_words == 0


@pytest.mark.parametrize(
    ("stride", "degree"),
    [(1, 1), (2, 2), (3, 1), (4, 4), (8, 8), (16, 16), (32, 32), (33, 1), (64, 32)],
)
def test_word_stride_conflicts_follow_shared_bank_modulo_arithmetic(stride, degree):
    analysis = analyze_banks(strided_access(stride_elements=stride))

    assert analysis.conflict_degree == degree
    assert analysis.banks_used == 32 // degree


def test_broadcast_read_is_one_conflict_free_multicast():
    analysis = analyze_banks(broadcast_access(address_bytes=20))

    assert analysis.conflict_degree == 1
    assert analysis.distinct_words == 1
    assert analysis.multicast_words == 1
    assert analysis.bank_words[5] == (5,)


def test_irregular_accesses_from_the_programming_guide_are_conflict_free():
    others = iter(bank for bank in range(32) if bank != 5)
    same_word_lanes = {3, 4, 6, 7, 9}
    shared = words(
        *(5 if lane in same_word_lanes else next(others) for lane in range(32))
    )

    assert analyze_banks(permuted_access(seed=74)).conflict_free
    assert analyze_banks(shared).conflict_free
    assert analyze_banks(shared).multicast_words == 1


def test_sub_word_lanes_inside_one_word_share_it_without_conflict():
    bytes_view = analyze_banks(contiguous_access(access_bytes=1))
    halves = analyze_banks(contiguous_access(access_bytes=2))

    assert (bytes_view.distinct_words, bytes_view.multicast_words) == (8, 8)
    assert bytes_view.conflict_degree == 1
    assert (halves.distinct_words, halves.multicast_words) == (16, 16)
    assert halves.conflict_degree == 1


def test_sub_word_accesses_to_different_words_of_one_bank_conflict():
    mixed = WarpAccess(1, (0, 1, 2, 3, 128) + (None,) * 27)
    halves = strided_access(stride_elements=64, access_bytes=2)

    assert analyze_banks(mixed).conflict_degree == 2
    assert analyze_banks(mixed).bank_words[0] == (0, 32)
    assert analyze_banks(halves).conflict_degree == 32


def test_two_words_in_one_bank_serialize_into_two_wavefronts():
    analysis = analyze_banks(words(0, 32, *range(1, 31)))

    assert analysis.conflict_degree == 2
    assert analysis.wavefronts == 2
    assert analysis.bank_words[0] == (0, 32)


def test_inactive_lanes_do_not_contribute_conflicts():
    partial = strided_access(stride_elements=32).masked(lane_mask(range(4)))

    assert analyze_banks(partial).conflict_degree == 4
    assert analyze_banks(partial).active_lanes == 4


@pytest.mark.parametrize("width", [8, 16])
def test_eight_and_sixteen_byte_accesses_are_an_explicit_exclusion(width):
    with pytest.raises(MemoryAccessError, match="excluded") as error:
        analyze_banks(contiguous_access(access_bytes=width))

    assert str(error.value) == BANK_EXCLUSION


def test_summary_counts_serialized_wavefronts_over_requests():
    summary = summarize_banks(
        (
            analyze_banks(strided_access(stride_elements=32)),
            analyze_banks(contiguous_access()),
        )
    )

    assert summary.requests == 2
    assert summary.wavefronts == 33
    assert summary.excess_wavefronts == 31
    assert summary.max_conflict_degree == 32
    assert summary.wavefronts_per_request == 16.5
    assert summary.to_mapping()["excess_wavefronts"] == 31


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: analyze_banks(None), TypeError),
        (lambda: summarize_banks(()), MemoryAccessError),
        (
            lambda: summarize_banks([analyze_banks(contiguous_access())]),
            MemoryAccessError,
        ),
        (lambda: summarize_banks((object(),)), TypeError),
    ],
)
def test_bank_contracts_reject_invalid_inputs(call, error):
    with pytest.raises(error):
        call()


def test_bank_mapping_exposes_degree_and_wavefronts():
    mapping = analyze_banks(strided_access(stride_elements=2)).to_mapping()

    assert mapping == {
        "access_bytes": 4,
        "active_lanes": 32,
        "distinct_words": 32,
        "multicast_words": 0,
        "banks_used": 16,
        "conflict_degree": 2,
        "wavefronts": 2,
    }
