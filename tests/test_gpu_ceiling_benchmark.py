import itertools

import numpy
import pytest

from ai_journey.gpu_ceiling_benchmark import (
    BankPoint,
    FmaPoint,
    StreamPoint,
    bank_accesses_per_call,
    benchmark_banks,
    benchmark_fma,
    benchmark_streams,
    fma_flops_per_call,
    stream_bytes_per_call,
    time_rotation,
)
from ai_journey.gpu_ceiling_reference import copy_source
from ai_journey.gpu_ceiling_runtime import KernelWorkload
from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.kernel_metal_benchmark import DeviceTiming, RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_reference import array_sha256
from test_gpu_ceiling_backend import NumpyBackend
from test_gpu_ceiling_runtime import ceiling_runtime
from test_kernel_metal_runtime import requires_metal


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


class Log:
    """Record each launch, synchronize and clock reading in one event list."""

    def __init__(self):
        self.events = []

    def workload(self, name):
        return KernelWorkload(
            run=lambda: self.events.append(name),
            synchronize=lambda: self.events.append("sync"),
            output=lambda: None,
        )

    def clock(self, step=100):
        counter = itertools.count(0, step)

        def read():
            self.events.append("clock")
            return next(counter)

        return read


def test_samples_rotate_over_the_points_and_reverse_every_repeat():
    log = Log()
    workloads = (log.workload("a"), log.workload("b"), log.workload("c"))

    timings = time_rotation(
        workloads,
        labels=("a", "b", "c"),
        calls=(1, 2, 1),
        warmup=1,
        repeats=3,
        clock=log.clock(),
    )

    warm = ["sync", "a", "sync", "sync", "b", "b", "sync", "sync", "c", "sync"]
    assert log.events[: len(warm)] == warm

    def timed(name, calls):
        return ["sync", "clock", *[name] * calls, "sync", "clock"]

    batches = {"a": 1, "b": 2, "c": 1}
    order = ["a", "b", "c", "c", "b", "a", "a", "b", "c"]
    expected = [event for name in order for event in timed(name, batches[name])]
    assert log.events[len(warm) :] == expected
    assert timings[1] == DeviceTiming("b", 2, 1, 3, (100, 100, 100), "injected")
    assert [timing.implementation for timing in timings] == ["a", "b", "c"]


def test_the_rotation_refuses_bad_sweeps_and_a_stalled_clock():
    log = Log()
    one = (log.workload("a"),)

    for arguments, message in (
        ({"labels": ("a", "b"), "calls": (1,)}, "a label and calls"),
        ({"labels": ("a",), "calls": (0,)}, "calls_per_sample must be in"),
        ({"labels": ("a",), "calls": (1,), "repeats": 2}, "repeats must be in"),
    ):
        settings = {"warmup": 0, "repeats": 3, **arguments}
        with pytest.raises(GpuCeilingError, match=message):
            time_rotation(one, clock=scripted_clock(), **settings)
    with pytest.raises(GpuCeilingError, match="did not advance"):
        time_rotation(
            one, labels=("a",), calls=(1,), warmup=0, repeats=3, clock=lambda: 7
        )


def test_the_run_budget_stops_a_sweep_between_samples():
    log = Log()
    clock = scripted_clock(10**9)
    budget = RunBudget(2, clock)

    with pytest.raises(KernelMetalError, match="max_runtime_seconds=2"):
        time_rotation(
            (log.workload("a"),),
            labels=("a",),
            calls=(1,),
            warmup=0,
            repeats=5,
            clock=clock,
            budget=budget,
        )


def test_work_is_counted_as_flops_compulsory_bytes_and_loads():
    assert fma_flops_per_call(1 << 20, 16, 8192) == 2 * (1 << 20) * 16 * 8192
    assert stream_bytes_per_call("stream_copy", 1 << 24, 4) == 2 * (1 << 28)
    # 4099 float4 values in walks of 16: 257 partials of 4 bytes each.
    assert stream_bytes_per_call("stream_read_reduce", 4099, 16) == 16 * 4099 + 4 * 257
    assert bank_accesses_per_call(1 << 20, 256) == (1 << 20) * 256 * 16
    with pytest.raises(GpuCeilingError, match="stream kernel"):
        stream_bytes_per_call("vector_add", 16, 1)


def test_the_fma_sweep_times_every_chain_and_step_count_in_order():
    backend = NumpyBackend()

    points = benchmark_fma(
        backend,
        seed=74,
        threads=1000,
        group_size=256,
        chain_counts=(1, 4),
        fma_steps=(8, 32),
        warmup=1,
        repeats=3,
        clock=scripted_clock(),
    )

    assert [(point.chains, point.fma_steps) for point in points] == [
        (1, 8),
        (1, 32),
        (4, 8),
        (4, 32),
    ]
    assert all(isinstance(point, FmaPoint) for point in points)
    assert backend.calls == 4 * (1 + 3)
    mapping = points[3].to_mapping()
    assert mapping["kernel"] == "fma_chains_4"
    assert mapping["flops_per_call"] == 2 * 1000 * 4 * 32
    assert mapping["samples_ns"] == [1000, 1000, 1000]
    assert mapping["median_gflops_per_second"] == 256000 / 1000
    assert mapping["best_gflops_per_second"] == 256.0
    assert mapping["launch"]["padding_threads"] == 24


def test_the_stream_sweep_batches_calls_to_the_byte_target():
    backend = NumpyBackend()

    points = benchmark_streams(
        backend,
        seed=74,
        group_size=256,
        float4_counts=(1024, 4096),
        copy_float4_per_thread=4,
        reduce_float4_per_thread=16,
        target_bytes_per_sample=1 << 18,
        warmup=0,
        repeats=3,
        clock=scripted_clock(),
    )

    assert [(point.kernel, point.float4_count) for point in points] == [
        ("stream_copy", 1024),
        ("stream_copy", 4096),
        ("stream_read_reduce", 1024),
        ("stream_read_reduce", 4096),
    ]
    assert all(isinstance(point, StreamPoint) for point in points)
    copy = points[1].to_mapping()
    assert (copy["buffer_bytes"], copy["bytes_per_call"]) == (65536, 131072)
    assert copy["calls_per_sample"] == 2
    assert copy["median_gb_per_second"] == 262144 / 1000
    assert copy["output_sha256"] == array_sha256(copy_source(74, 1, 4096))
    reduce = points[2].to_mapping()
    assert reduce["threads"] == 64
    assert reduce["bytes_per_call"] == 16 * 1024 + 4 * 64
    assert reduce["calls_per_sample"] == 16


def test_the_bank_sweep_reports_time_per_access_for_each_layout():
    points = benchmark_banks(
        NumpyBackend(),
        threads=512,
        group_size=256,
        row_words=(32, 33),
        strides=(1, 4),
        trips=3,
        warmup=0,
        repeats=3,
        clock=scripted_clock(),
    )

    assert [(point.row_words, point.stride) for point in points] == [
        (32, 1),
        (32, 4),
        (33, 1),
        (33, 4),
    ]
    assert all(isinstance(point, BankPoint) for point in points)
    mapping = points[1].to_mapping()
    assert mapping["accesses_per_call"] == 512 * 3 * 16
    assert mapping["median_ps_per_access"] == 1000 * 1000 / (512 * 3 * 16)
    assert mapping["kernel"] == "bank_rows_32"


def test_a_timed_output_that_differs_from_its_rebuild_stops_the_sweep():
    class Drifting(NumpyBackend):
        def bank_workload(self, row_words, stride, trips, launch):
            workload = super().bank_workload(row_words, stride, trips, launch)
            wrong = workload.output() + numpy.uint32(1)
            return KernelWorkload(workload.run, workload.synchronize, lambda: wrong)

    with pytest.raises(GpuCeilingError, match="timed output differs"):
        benchmark_banks(
            Drifting(),
            threads=256,
            group_size=256,
            row_words=(32,),
            strides=(1,),
            trips=1,
            warmup=0,
            repeats=3,
            clock=scripted_clock(),
        )


@requires_metal
def test_each_sweep_runs_and_checks_its_outputs_on_this_gpu():
    runtime = ceiling_runtime()
    common = {"warmup": 0, "repeats": 3}

    fma = benchmark_fma(
        runtime,
        seed=74,
        threads=4096,
        group_size=256,
        chain_counts=(1, 2),
        fma_steps=(8, 16),
        **common,
    )
    streams = benchmark_streams(
        runtime,
        seed=74,
        group_size=256,
        float4_counts=(4096,),
        copy_float4_per_thread=4,
        reduce_float4_per_thread=16,
        target_bytes_per_sample=1 << 17,
        **common,
    )
    banks = benchmark_banks(
        runtime,
        threads=512,
        group_size=256,
        row_words=(32, 33),
        strides=(1, 32),
        trips=2,
        **common,
    )

    for point in (*fma, *streams, *banks):
        assert point.timing.timer == "time.perf_counter_ns"
        assert all(sample > 0 for sample in point.timing.samples_ns)
