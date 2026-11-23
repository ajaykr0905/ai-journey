"""Check every scan and ``torch.cumsum`` on one seeded case, several runs each.

Each case draws its input from the seed and runs every admitted kernel and
the vendor side ``runs`` times, each run from a fresh output buffer. Three
rules stop the run when a kernel breaks one, because the output is wrong:

- the sentinel guard tail of every run keeps its bits;
- every element of every run stays within the kernel's bound;
- on a ``small_integers`` case, every run equals the exact prefix sums.

Everything else is a measured property and never stops the run:

- ``deterministic``: every run of a side gave the same bits;
- ``replay_bits``: the first run of S0 or S1 holds the bits of its CPU replay,
  with the count of elements that differ; S2 has no replay, so ``None``;
- ``exact_bits`` of ``torch.cumsum`` on an integer case, and its largest
  relative error. PyTorch does not report the order of ``torch.cumsum``, so it
  has no declared bound, and only a value that is not finite stops the run.

The records keep the digest of every run, so a verifier can rebuild each
output that holds the replay's or the exact bits, with its tail.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy

from .gpu_scan_plan import VENDOR
from .gpu_scan_reference import (
    CORRECTNESS_STREAM,
    INTEGERS,
    REPLAYS,
    error_bound,
    exact_prefix,
    float64_reference,
    float64_sha256,
    max_relative_error,
    scan_inputs,
)
from .gpu_scan_runtime import admitted_names, scan_plan
from .gpu_scan_source import KERNELS, GpuScanError, guard_elements, level_lengths
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, array_sha256

MIN_RUNS = 2
MAX_RUNS = 5


def runnable(backend: object, names: Iterable[str]) -> tuple[str, ...]:
    """Keep the registered kernels that the backend's pipelines admitted, in order."""

    chosen = tuple(names)
    for name in chosen:
        if name not in KERNELS:
            raise GpuScanError(f"{name!r} is not a registered scan kernel")
    allowed = admitted_names(backend.admissions)
    return tuple(name for name in KERNELS if name in chosen and name in allowed)


@dataclass(frozen=True)
class SideOutput:
    """What every run of one side gave on one case."""

    side: str
    run_sha256: tuple[str, ...]
    guard_elements: int | None
    max_relative_error: float
    bound: float | None
    replay_bits: bool | None
    replay_differences: int | None
    exact_bits: bool | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_relative_error < math.inf:
            raise GpuScanError("a recorded error must be finite")
        if (self.replay_bits is None) != (self.replay_differences is None):
            raise GpuScanError("replay bits and their count of differences go together")

    @property
    def deterministic(self) -> bool:
        """Whether every run gave the same bits."""

        return len(set(self.run_sha256)) == 1

    def to_mapping(self) -> dict[str, object]:
        """Return the side's record as a report keeps it."""

        return {
            "side": self.side,
            "buffer_sha256": self.run_sha256[0],
            "run_sha256": list(self.run_sha256),
            "deterministic": self.deterministic,
            "guard_elements": self.guard_elements,
            "max_relative_error": self.max_relative_error,
            "bound": self.bound,
            "bound_ratio": None
            if self.bound is None
            else self.max_relative_error / self.bound,
            "replay_bits": self.replay_bits,
            "replay_differences": self.replay_differences,
            "exact_bits": self.exact_bits,
        }


@dataclass(frozen=True)
class ScanCase:
    """One seeded case: its input digests, every launch and every side's runs."""

    index: int
    length: int
    pattern: str
    launches: dict[str, list[dict[str, object]]]
    input_sha256: str
    reference_sha256: str
    replay_sha256: dict[str, str]
    exact_sha256: str | None
    outputs: tuple[SideOutput, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report records it."""

        return {
            "index": self.index,
            "length": self.length,
            "pattern": self.pattern,
            "levels": list(level_lengths(self.length)),
            "launches": self.launches,
            "input_sha256": self.input_sha256,
            "reference_sha256": self.reference_sha256,
            "replay_sha256": self.replay_sha256,
            "exact_sha256": self.exact_sha256,
            "outputs": [output.to_mapping() for output in self.outputs],
        }


def _same_bits(first: numpy.ndarray, second: numpy.ndarray) -> bool:
    return first.shape == second.shape and numpy.array_equal(
        first.view(numpy.uint32), second.view(numpy.uint32)
    )


def _kernel_runs(
    backend: object,
    name: str,
    values: numpy.ndarray,
    runs: int,
    label: str,
) -> list[numpy.ndarray]:
    """Run one kernel ``runs`` times; every run must keep its whole guard tail."""

    length = values.size
    guard = guard_elements(length)
    buffers = []
    for _ in range(runs):
        buffer = backend.scan(name, values)
        if (
            not isinstance(buffer, numpy.ndarray)
            or buffer.dtype != numpy.float32
            or buffer.shape != (length + guard,)
        ):
            raise GpuScanError(f"{label}: {name} returned another buffer shape")
        if not (buffer[length:].view(numpy.uint32) == FLOAT32_SENTINEL_BITS).all():
            raise GpuScanError(f"{label}: {name} wrote into its guard tail")
        buffers.append(buffer)
    return buffers


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    length: int,
    pattern: str,
    names: tuple[str, ...],
    runs: int,
) -> ScanCase:
    """Check every admitted kernel and the vendor side on one case."""

    if type(runs) is not int or not MIN_RUNS <= runs <= MAX_RUNS:
        raise GpuScanError(f"a case runs each side {MIN_RUNS} to {MAX_RUNS} times")
    names = runnable(backend, names)
    values = scan_inputs(seed, CORRECTNESS_STREAM, index, length, pattern)
    reference, scale = float64_reference(values)
    exact = exact_prefix(values) if pattern == INTEGERS else None
    replays = {name: REPLAYS[name](values) for name in names if name in REPLAYS}
    label = f"case {index} ({length} {pattern})"
    outputs = []
    for name in names:
        buffers = _kernel_runs(backend, name, values, runs, label)
        bound = error_bound(name, length)
        errors = [
            max_relative_error(item[:length], reference, scale) for item in buffers
        ]
        if not max(errors) <= bound:
            raise GpuScanError(
                f"{label}: {name} has relative error {max(errors)!r}, above its "
                f"bound {bound!r}"
            )
        if exact is not None and not all(
            _same_bits(item[:length], exact) for item in buffers
        ):
            raise GpuScanError(f"{label}: {name} missed the exact integer sums")
        replay = replays.get(name)
        differences = (
            None
            if replay is None
            else int(
                numpy.count_nonzero(
                    buffers[0][:length].view(numpy.uint32) != replay.view(numpy.uint32)
                )
            )
        )
        outputs.append(
            SideOutput(
                side=name,
                run_sha256=tuple(array_sha256(item) for item in buffers),
                guard_elements=guard_elements(length),
                max_relative_error=max(errors),
                bound=bound,
                replay_bits=None if replay is None else differences == 0,
                replay_differences=differences,
                exact_bits=None if exact is None else True,
            )
        )
    vendor = []
    for _ in range(runs):
        result = backend.cumsum(values)
        if (
            not isinstance(result, numpy.ndarray)
            or result.dtype != numpy.float32
            or result.shape != (length,)
        ):
            raise GpuScanError(f"{label}: torch.cumsum returned another shape")
        vendor.append(result)
    errors = [max_relative_error(item, reference, scale) for item in vendor]
    if not max(errors) < math.inf:
        raise GpuScanError(f"{label}: torch.cumsum gave a value that is not finite")
    outputs.append(
        SideOutput(
            side=VENDOR,
            run_sha256=tuple(array_sha256(item) for item in vendor),
            guard_elements=None,
            max_relative_error=max(errors),
            bound=None,
            replay_bits=None,
            replay_differences=None,
            exact_bits=None
            if exact is None
            else all(_same_bits(item, exact) for item in vendor),
        )
    )
    return ScanCase(
        index=index,
        length=length,
        pattern=pattern,
        launches={
            name: [
                level.to_mapping() for level in scan_plan(name, backend.limits, length)
            ]
            for name in names
        },
        input_sha256=array_sha256(values),
        reference_sha256=float64_sha256(reference),
        replay_sha256={name: array_sha256(replay) for name, replay in replays.items()},
        exact_sha256=None if exact is None else array_sha256(exact),
        outputs=tuple(outputs),
    )
