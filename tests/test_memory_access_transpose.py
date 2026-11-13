import pytest

from ai_journey.memory_access_coalescing import analyze_coalescing, summarize_coalescing
from ai_journey.memory_access_pattern import (
    MatrixLayout,
    MemoryAccessError,
    column_access,
    row_access,
)
from ai_journey.memory_access_transpose import analyze_tile_transpose


def test_unpadded_float32_tile_column_reads_are_thirty_two_way_conflicted():
    analysis = analyze_tile_transpose(leading_dimension=32, matrix_size=1024)

    reads = analysis.shared_column_reads
    assert (reads.requests, reads.max_conflict_degree) == (32, 32)
    assert reads.wavefronts == 1024
    assert reads.excess_wavefronts == 992
    assert analysis.shared_row_writes.max_conflict_degree == 1
    assert analysis.shared_row_writes.wavefronts == 32
    assert (analysis.tile_bytes, analysis.padding_bytes) == (4096, 0)


def test_padding_the_leading_dimension_to_33_removes_every_conflict():
    analysis = analyze_tile_transpose(leading_dimension=33, matrix_size=1024)

    reads = analysis.shared_column_reads
    assert (reads.max_conflict_degree, reads.wavefronts) == (1, 32)
    assert reads.excess_wavefronts == 0
    assert analysis.shared_row_writes.max_conflict_degree == 1
    assert (analysis.tile_bytes, analysis.padding_bytes) == (4224, 128)


@pytest.mark.parametrize(
    ("leading_dimension", "degree"),
    [(34, 2), (36, 4), (40, 8), (48, 16), (64, 32), (65, 1)],
)
def test_float32_column_conflicts_equal_the_bank_stride_common_divisor(
    leading_dimension, degree
):
    analysis = analyze_tile_transpose(
        leading_dimension=leading_dimension, matrix_size=128
    )

    assert analysis.shared_column_reads.max_conflict_degree == degree
    assert analysis.shared_column_reads.wavefronts == 32 * degree


@pytest.mark.parametrize(
    ("element_bytes", "leading_dimension", "wavefronts", "degree"),
    [
        (2, 32, 512, 16),
        (2, 33, 48, 2),
        (2, 34, 32, 1),
        (1, 32, 256, 8),
        (1, 33, 56, 2),
        (1, 36, 32, 1),
    ],
)
def test_narrow_tiles_need_one_full_bank_word_of_padding(
    element_bytes, leading_dimension, wavefronts, degree
):
    analysis = analyze_tile_transpose(
        leading_dimension=leading_dimension,
        element_bytes=element_bytes,
        matrix_size=128,
    )

    assert analysis.shared_column_reads.wavefronts == wavefronts
    assert analysis.shared_column_reads.max_conflict_degree == degree


def test_naive_transpose_stores_columns_while_tiled_stores_rows():
    analysis = analyze_tile_transpose(leading_dimension=33, matrix_size=1024)

    assert analysis.tiles == 1024
    assert analysis.naive_loads.requests == 32_768
    assert analysis.naive_loads.sectors_per_request == 4
    assert analysis.naive_stores.sectors_per_request == 32
    assert analysis.naive_stores.efficiency == 0.125
    assert analysis.tiled_stores.sectors_per_request == 4
    assert analysis.tiled_stores.efficiency == 1.0
    assert analysis.naive_stores.transferred_bytes == 8 * (
        analysis.tiled_stores.transferred_bytes
    )
    assert analysis.tiled_loads == analysis.naive_loads


@pytest.mark.parametrize("element_bytes", [1, 2, 4])
def test_representative_global_requests_match_full_enumeration(element_bytes):
    size = 128
    analysis = analyze_tile_transpose(
        leading_dimension=33, element_bytes=element_bytes, matrix_size=size
    )
    matrix = MatrixLayout(
        rows=size, columns=size, leading_dimension=size, element_bytes=element_bytes
    )
    rows = summarize_coalescing(
        tuple(
            analyze_coalescing(row_access(matrix, row=row, start_column=start))
            for row in range(size)
            for start in range(0, size, 32)
        )
    )
    columns = summarize_coalescing(
        tuple(
            analyze_coalescing(column_access(matrix, column=column, start_row=start))
            for column in range(size)
            for start in range(0, size, 32)
        )
    )

    assert analysis.naive_loads == rows
    assert analysis.tiled_stores == rows
    assert analysis.naive_stores == columns


def test_transpose_mapping_names_per_tile_and_per_matrix_sections():
    mapping = analyze_tile_transpose(matrix_size=256).to_mapping()

    assert mapping["tile"] == 32
    assert mapping["tiles"] == 64
    assert mapping["shared_column_reads_per_tile"]["max_conflict_degree"] == 32
    assert mapping["naive_global_stores"]["requests"] == 2048
    assert mapping["tiled_global_stores"]["sectors_per_request"] == 4


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"matrix_size": 96}, MemoryAccessError, "multiple of 128"),
        ({"matrix_size": 0}, MemoryAccessError, "multiple of 128"),
        ({"matrix_size": 128.0}, TypeError, "matrix_size"),
        ({"matrix_size": 128, "leading_dimension": 31}, MemoryAccessError, "32"),
        ({"matrix_size": 128, "leading_dimension": True}, TypeError, "leading"),
        ({"matrix_size": 128, "element_bytes": 8}, MemoryAccessError, "excluded"),
        ({"matrix_size": 128, "element_bytes": 3}, MemoryAccessError, "1, 2, 4"),
    ],
)
def test_transpose_rejects_unsupported_geometry(arguments, error, message):
    with pytest.raises(error, match=message):
        analyze_tile_transpose(**arguments)
