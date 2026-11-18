"""Unprofiled wall-clock timing of the pinned workloads on this host's CPU.

Each workload runs ``warmup`` untimed passes, then ``repeats`` timed passes
through ``memory_access_benchmark.time_repeats``. Every pass, timed or not,
must reproduce its reference digest, and that check runs after the clock
stops. The statistics are that module's nearest-rank median and IQR, with
no interpolation. No profiler runs while the clock runs, so these numbers,
not the profiler's, give each round's speedup.

Only the builtin ``time.perf_counter_ns`` makes a measurement; any other
clock is ``injected`` and its durations are synthetic. The 1, 5 and 15
minute load averages come with every run, because other work on the host
slows a pure-Python loop. These are host-specific CPU measurements.
"""

from __future__ import annotations

import math
import os
import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .kernel_launch_config import require_int
from .memory_access_benchmark import (
    TimingSummary,
    summarize_timings,
    time_repeats,
    timer_name,
)
from .sim_profile_config import MAX_REPEATS, MAX_WARMUP, MIN_REPEATS, SimProfileError
from .sim_profile_workloads import Workload

LOAD_AVERAGE_SOURCE = (
    "os.getloadavg(): the 1, 5 and 15 minute load averages that uptime prints, "
    "read before the first profile and after the last timed pass"
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,39}")


@dataclass(frozen=True)
class WorkloadTiming:
    """The raw timed passes of one workload, with the clock that took them."""

    workload: str
    warmup: int
    repeats: int
    samples_ns: tuple[int, ...]
    timer: str

    @property
    def summary(self) -> TimingSummary:
        return summarize_timings(self.samples_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples beside their statistics, so both can be checked."""

        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "samples_ns": list(self.samples_ns),
            "timing": self.summary.to_mapping(),
        }


def time_workload(
    workload: Workload,
    *,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> WorkloadTiming:
    """Time ``repeats`` checked passes of one workload after untimed warmups."""

    if type(workload) is not Workload:
        raise TypeError("workload must be a Workload")
    require_int(warmup, "warmup", minimum=0, maximum=MAX_WARMUP)
    require_int(repeats, "repeats", minimum=MIN_REPEATS, maximum=MAX_REPEATS)
    try:
        samples = time_repeats(
            workload.run,
            warmup=warmup,
            repeats=repeats,
            clock=clock,
            inspect=workload.check,
        )
    except SimProfileError:
        raise
    except ValueError as error:
        raise SimProfileError(f"timing {workload.name} failed: {error}") from error
    return WorkloadTiming(
        workload=workload.name,
        warmup=warmup,
        repeats=repeats,
        samples_ns=samples,
        timer=timer_name(clock),
    )


def read_load_average(
    reader: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> list[float]:
    """Read the three load averages; refuse anything but three finite floats."""

    values = reader()
    if (
        type(values) is not tuple
        or len(values) != 3
        or not all(
            type(value) is float and math.isfinite(value) and value >= 0
            for value in values
        )
    ):
        raise SimProfileError("the load average must be three nonnegative floats")
    return list(values)


def _token(value: object) -> str:
    if type(value) is str and _TOKEN.fullmatch(value):
        return value
    return "unknown"


def host_environment() -> dict[str, str]:
    """Name the interpreter and platform without naming the host or its user.

    Only short version and platform tokens are kept; anything else becomes
    ``unknown``. The CPU model is not recorded.
    """

    return {
        "python_version": _token(platform.python_version()),
        "python_implementation": _token(platform.python_implementation()),
        "platform_system": _token(platform.system()),
        "platform_machine": _token(platform.machine()),
    }
