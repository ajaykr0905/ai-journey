"""Tests of the declared memory and traffic model, and one CPU allocation check."""

import tracemalloc

import pytest

from ai_journey.triton_flash_kernel import TritonFlashError
from ai_journey.triton_flash_memory import (
    MEMORY_TABLE,
    MODEL_HEAD_DIMS,
    SEQUENCE_LENGTHS,
    STATEMENT,
    MemoryRow,
    mebibytes,
    memory_mapping,
)
from ai_journey.triton_flash_reference import (
    AttentionCase,
    attention_inputs,
    blockwise_attention,
    naive_float32,
)


def test_the_table_covers_every_declared_length_and_head_dim():
    assert [(row.sequence, row.head_dim) for row in MEMORY_TABLE] == [
        (n, d) for d in MODEL_HEAD_DIMS for n in SEQUENCE_LENGTHS
    ]
    assert SEQUENCE_LENGTHS == (512, 1024, 2048, 4096, 8192)
    assert MODEL_HEAD_DIMS == (64, 128)


def test_the_rule_for_one_row_by_hand():
    row = MemoryRow(4096, 64).to_mapping()

    # S and P: 2 * 4096 * 4096 * 4 bytes = 128 MiB.
    assert row["naive_intermediate_bytes"] == 128 << 20
    # Q, K, V and O: 4 * 4096 * 64 * 4 bytes = 4 MiB.
    assert row["io_bytes"] == 4 << 20
    assert row["naive_bytes"] == 132 << 20
    # m and l: 2 * 4096 * 4 bytes = 32 KiB.
    assert row["flash_intermediate_bytes"] == 32 << 10
    assert row["flash_bytes"] == (4 << 20) + (32 << 10)
    assert row["memory_ratio"] == (132 << 20) / ((4 << 20) + (32 << 10))
    assert row["naive_traffic_bytes"] == (4 * 4096 * 4096 + 4 * 4096 * 64) * 4
    assert (
        row["flash_traffic_bytes"]
        == (2 * 4096 * 64 + 2 * 4096 * 64 * 32 + 2 * 4096) * 4
    )


def test_the_memory_gap_grows_with_n_and_the_traffic_gap_tends_to_2_block_m_over_d():
    for dim in MODEL_HEAD_DIMS:
        rows = [row.to_mapping() for row in MEMORY_TABLE if row.head_dim == dim]
        ratios = [row["memory_ratio"] for row in rows]
        assert ratios == sorted(ratios) and ratios[-1] > 30
        assert all(row["traffic_ratio"] > 1 for row in rows)
        assert rows[-1]["traffic_ratio"] == pytest.approx(2 * 128 / dim, rel=0.02)
    largest = MemoryRow(8192, 128)
    assert mebibytes(largest.naive_intermediate_bytes) == "512.00"


def test_the_mapping_says_it_is_a_model_and_not_a_measurement():
    mapping = memory_mapping()

    assert mapping["measured"] is False
    assert mapping["statement"] == STATEMENT
    assert "not a measured GPU footprint" in STATEMENT
    assert len(mapping["rows"]) == 10


def test_a_row_refuses_a_bad_size():
    with pytest.raises(TritonFlashError):
        MemoryRow(0, 64)
    with pytest.raises(TypeError):
        MemoryRow(512.0, 64)


def peak(function):
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_numpy_on_the_cpu_shows_the_n_by_n_buffer_only_in_the_naive_form():
    # A host CPU allocation check, not a GPU number. The NumPy naive form
    # reuses S for P in place, so it holds one N x N array, not two.
    case = AttentionCase(1, 1024, 64, False)
    q, k, v = attention_inputs("normal", 74, 0, case)
    square = 1024 * 1024 * 4

    naive = peak(lambda: naive_float32(q, k, v, False))
    flash = peak(lambda: blockwise_attention(q, k, v, False, 64, 64))

    assert naive >= square
    assert flash < square // 4
