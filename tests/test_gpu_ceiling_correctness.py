import numpy
import pytest

from ai_journey.gpu_ceiling_correctness import (
    BankCase,
    FmaCase,
    StreamCase,
    check_bank,
    check_fma,
    check_stream,
)
from ai_journey.gpu_ceiling_reference import (
    FMA_ADDEND,
    FMA_MULTIPLIER,
    copy_source,
    fma_lane_words,
    fma_start_values,
    lane_tile,
    reduce_source,
    words_sha256,
)
from ai_journey.gpu_ceiling_source import FMA_CHAIN_COUNTS, GpuCeilingError
from ai_journey.kernel_metal_reference import array_sha256, float32_sentinels
from test_gpu_ceiling_backend import NumpyBackend
from test_gpu_ceiling_runtime import ceiling_runtime
from test_kernel_metal_runtime import requires_metal


class Corrupting(NumpyBackend):
    """Damage one output in a chosen way, as a broken kernel would."""

    def __init__(self, damage):
        super().__init__()
        self.damage = damage

    def fma(self, chains, start, launch, steps):
        return self.damage(super().fma(chains, start, launch, steps))

    def stream(self, name, source, launch, **arguments):
        return self.damage(super().stream(name, source, launch, **arguments))

    def bank(self, row_words, stride, trips, launch):
        return self.damage(super().bank(row_words, stride, trips, launch))


def flip_first(buffer):
    changed = buffer.copy()
    words = changed.view(numpy.uint32)
    words[0] ^= 1
    return changed


def write_tail(buffer):
    changed = buffer.copy()
    changed.view(numpy.uint32)[-1] = 0
    return changed


class Unfused(NumpyBackend):
    """Round the product and the sum separately: a kernel without a real FMA."""

    def fma(self, chains, start, launch, steps):
        values = start.copy()
        for _ in range(steps):
            values = (values * FMA_MULTIPLIER).astype(numpy.float32) + FMA_ADDEND
        words = lane_tile(fma_lane_words(values, chains), launch.geometry.extent[0])
        tail = float32_sentinels(launch.geometry.padding_threads).view(numpy.uint32)
        return numpy.concatenate([words, tail])


FMA_CASE = {
    "seed": 74,
    "chains": 4,
    "threads": 1000,
    "group_size": 256,
    "fma_steps": 64,
}
COPY_CASE = {
    "seed": 74,
    "kernel": "stream_copy",
    "float4_count": 4099,
    "float4_per_thread": 4,
    "group_size": 256,
}
REDUCE_CASE = {**COPY_CASE, "kernel": "stream_read_reduce", "float4_per_thread": 16}
BANK_CASE = {
    "row_words": 33,
    "stride": 8,
    "threads": 1000,
    "group_size": 256,
    "trips": 9,
}


def test_an_fma_case_records_digests_that_rebuild_from_the_seed():
    case = check_fma(NumpyBackend(), **FMA_CASE)

    assert isinstance(case, FmaCase)
    mapping = case.to_mapping()
    assert (mapping["kernel"], mapping["chains"], mapping["fma_steps"]) == (
        "fma_chains_4",
        4,
        64,
    )
    assert (mapping["threads"], mapping["guard_words"]) == (1000, 24)
    assert mapping["launch"]["padding_threads"] == 24
    assert mapping["start_sha256"] == array_sha256(fma_start_values(74))
    buffer = NumpyBackend().fma(4, fma_start_values(74), case.launch, 64)
    assert mapping["buffer_sha256"] == words_sha256(buffer)
    assert mapping["expected_sha256"] == words_sha256(buffer[:1000])


def test_stream_cases_rebuild_the_copy_and_the_exact_partials():
    copy = check_stream(NumpyBackend(), **COPY_CASE).to_mapping()
    reduce = check_stream(NumpyBackend(), **REDUCE_CASE).to_mapping()

    assert copy["threads"] == 1025 and copy["guard_words"] == 128
    assert copy["source_sha256"] == array_sha256(copy_source(74, 0, 4099))
    assert copy["expected_sha256"] == copy["source_sha256"]
    assert reduce["threads"] == 257 and reduce["guard_words"] == 255
    assert reduce["source_sha256"] == array_sha256(reduce_source(74, 0, 4099))
    assert isinstance(check_stream(NumpyBackend(), **COPY_CASE), StreamCase)


def test_a_bank_case_records_its_stride_layout_and_digests():
    case = check_bank(NumpyBackend(), **BANK_CASE)

    assert isinstance(case, BankCase)
    mapping = case.to_mapping()
    assert (mapping["kernel"], mapping["stride"], mapping["trips"]) == (
        "bank_rows_33",
        8,
        9,
    )
    assert mapping["guard_words"] == 24


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (flip_first, "differs from"),
        (write_tail, "wrote into the guard tail"),
        (lambda buffer: buffer[:-1], "returned no"),
        (lambda buffer: buffer.astype(numpy.float64), "returned no"),
    ],
)
def test_every_check_stops_on_a_wrong_word_a_written_tail_or_a_bad_buffer(
    damage, message
):
    backend = Corrupting(damage)

    for check, case in (
        (check_fma, FMA_CASE),
        (check_stream, COPY_CASE),
        (check_stream, REDUCE_CASE),
        (check_bank, BANK_CASE),
    ):
        with pytest.raises(GpuCeilingError, match=message):
            check(backend, **case)


def test_the_fma_check_tells_an_unfused_kernel_from_a_fused_one():
    with pytest.raises(GpuCeilingError, match="differs from the exact FMA chains"):
        check_fma(Unfused(), **FMA_CASE)


def test_unknown_stream_kernels_are_refused():
    with pytest.raises(GpuCeilingError, match="stream_copy or stream_read_reduce"):
        check_stream(NumpyBackend(), **{**COPY_CASE, "kernel": "vector_add"})


@requires_metal
def test_every_kernel_passes_its_check_on_this_gpu():
    runtime = ceiling_runtime()

    for chains in FMA_CHAIN_COUNTS:
        check_fma(runtime, **{**FMA_CASE, "chains": chains})
    check_stream(runtime, **COPY_CASE)
    check_stream(runtime, **REDUCE_CASE)
    for row_words in (32, 33):
        for stride in (1, 2, 4, 8, 16, 32):
            check_bank(
                runtime, **{**BANK_CASE, "row_words": row_words, "stride": stride}
            )
