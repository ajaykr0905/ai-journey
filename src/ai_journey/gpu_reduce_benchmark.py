"""Time every reduction and ``torch.sum`` in one rotated run on one device buffer.

The run draws one seeded standard-normal input of the largest size and puts
it on the device once. At each size n it binds every side to the first n
values of that buffer: R0 to R5, each as pass 1 and the combine, then
``torch.sum(x[:n], 0, out=out)`` into a preallocated output. The timing is
the Day 74 and Day 76 method, run by the Day 74 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again.
3. Repeat r starts with side r mod 7 and runs each of the 7 sides once in
   order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose bytes, each element read once,
reach ``target_bytes_per_sample``, by the Day 73 ``calls_per_sample``. A
rate is decimal GB/s: the bytes of one sample over its median duration in
ns, with the range from the IQR of the samples. The partials that pass 2
reads are not counted.

After the last sample, the latest result of each side is checked against
the exact sum of its prefix: a kernel within the bound of its order, and
``torch.sum`` within the declared tolerance, or the run stops. Each side
then runs once more, and the record states whether the repeat kept the
bits. Only the host timer ``time.perf_counter_ns`` makes a measurement; any
other clock is named ``injected``. These are host-specific Apple GPU
(Metal) timings; nothing here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .gpu_reduce_bound import (
    VENDOR_DEPTH,
    ExactSums,
    ReductionCheck,
    benchmark_values,
    prefix_sums,
    summation_depth,
)
from .gpu_reduce_config import BYTES_PER_ELEMENT
from .gpu_reduce_plan import TORCH_SUM
from .gpu_reduce_runtime import SIDES, SideResult, checked_launches
from .gpu_reduce_source import STEP_KERNELS, GpuReduceError
from .kernel_metal_benchmark import (
    MAX_POINTS,
    DeviceTiming,
    RunBudget,
    _rates,
    calls_per_sample,
)
from .kernel_metal_geometry import CheckedLaunch
from .memory_access_benchmark import timer_name
from .metal_launch_sweep_benchmark import time_shapes

ORDER_BOUND = "order_bound"
DECLARED_TOLERANCE = "declared_tolerance"
GB_PER_SECOND = (
    "decimal GB/s: 4 * n bytes for each call, each element read once, times "
    "the calls of a sample, divided by the median sample duration in ns; the "
    "partials of pass 2 are not counted"
)


def rates(timing: DeviceTiming, bytes_per_call: int) -> dict[str, object]:
    """The Day 73 rates of one side, with the range from the IQR of its samples."""

    moved = bytes_per_call * timing.calls_per_sample
    summary = timing.summary
    return {
        **_rates(timing, bytes_per_call),
        "iqr_gb_per_second": [moved / summary.p75_ns, moved / summary.p25_ns],
    }


@dataclass(frozen=True)
class SideCheck:
    """One side's latest result against the exact sum of its prefix."""

    check: ReductionCheck
    rule: str
    same_bits_on_repeat: bool

    def to_mapping(self) -> dict[str, object]:
        """Return the check as a report records it."""

        return {
            **self.check.to_mapping(),
            "rule": self.rule,
            "same_bits_on_repeat": self.same_bits_on_repeat,
        }


def _checks(
    results: object, sums: ExactSums, n: int, label: str
) -> dict[str, SideCheck]:
    if not isinstance(results, Mapping) or tuple(results) != SIDES:
        raise GpuReduceError(f"{label}: the check must cover every side in order")
    checks = {}
    for name, result in results.items():
        if not isinstance(result, SideResult) or (
            type(result.same_bits_on_repeat) is not bool
        ):
            raise TypeError("a side result must be a SideResult")
        if name == TORCH_SUM:
            depth, rule = VENDOR_DEPTH, DECLARED_TOLERANCE
        else:
            depth, rule = summation_depth(name, n), ORDER_BOUND
        check = ReductionCheck(result.output, depth, sums)
        if not check.within:
            raise GpuReduceError(
                f"{label}: {name} sits {float(check.error)!r} from the exact sum; "
                f"its {rule.replace('_', ' ')} is {float(check.bound)!r}"
            )
        checks[name] = SideCheck(check, rule, result.same_bits_on_repeat)
    return checks


@dataclass(frozen=True)
class ReducePoint:
    """One timing size: every side timed on the same device buffer."""

    index: int
    n: int
    launches: Mapping[str, CheckedLaunch]
    timings: tuple[DeviceTiming, ...]
    sums: ExactSums
    checks: Mapping[str, SideCheck]

    @property
    def bytes_per_call(self) -> int:
        """Bytes that one call must read: each element once."""

        return BYTES_PER_ELEMENT * self.n

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: R0 to R5, then ``torch.sum``."""

        return tuple(timing.implementation for timing in self.timings)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics, rates and checks."""

        return {
            "index": self.index,
            "n": self.n,
            "bytes_per_call": self.bytes_per_call,
            "side_order": list(self.sides),
            "launches": {
                name: launch.to_mapping() for name, launch in self.launches.items()
            },
            "sides": {
                timing.implementation: rates(timing, self.bytes_per_call)
                for timing in self.timings
            },
            **self.sums.to_mapping(),
            "checks": {name: check.to_mapping() for name, check in self.checks.items()},
        }


def benchmark_point(
    backend: object,
    source: object,
    *,
    index: int,
    n: int,
    sums: ExactSums,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> ReducePoint:
    """Time every side at one size on the shared device buffer, then check it."""

    launches = {
        name: checked_launches(backend.limits, name, n)[0] for name in STEP_KERNELS
    }
    workload = backend.workload(source, n)
    if tuple(workload.sides) != SIDES:
        raise GpuReduceError("a workload must bind R0 to R5 and then torch.sum")
    calls = calls_per_sample(BYTES_PER_ELEMENT * n, target_bytes_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    checks = _checks(workload.check(), sums, n, f"point {index} (n={n})")
    timer = timer_name(clock)
    timings = tuple(
        DeviceTiming(name, calls, warmup, repeats, series, timer)
        for name, series in zip(SIDES, samples)
    )
    return ReducePoint(index, n, launches, timings, sums, checks)


def benchmark_sweep(
    backend: object,
    *,
    seed: int,
    sizes: tuple[int, ...],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[ReducePoint, ...]:
    """Upload one input of the largest size and time every side at every size."""

    if type(sizes) is not tuple or not 1 <= len(sizes) <= MAX_POINTS:
        raise GpuReduceError(f"a sweep holds 1 to {MAX_POINTS} sizes")
    values = benchmark_values(seed, max(sizes))
    sums = prefix_sums(values, sizes)
    source = backend.upload(values)
    return tuple(
        benchmark_point(
            backend,
            source,
            index=index,
            n=n,
            sums=sums[n],
            warmup=warmup,
            repeats=repeats,
            target_bytes_per_sample=target_bytes_per_sample,
            clock=clock,
            budget=budget,
        )
        for index, n in enumerate(sizes)
    )
