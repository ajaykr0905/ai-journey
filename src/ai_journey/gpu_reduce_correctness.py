"""Check every reduction kernel against the exact sum on seeded cases.

Each case draws its input from the seed, sums it exactly on the CPU and runs
both passes of every step kernel ``runs`` times on fresh guarded buffers.
These rules apply, and a case that breaks one stops the run:

- every partial slot of the launch holds a finite value, so every group
  wrote its partial;
- the tail past the partials and the tail past the output keep their
  quiet-NaN sentinels, so no thread wrote past its slot;
- the result is finite, because a read past n would add a sentinel of the
  input tail;
- the result lies within the bound of the kernel's order, and it equals the
  exact sum when every partial sum of every order is exact.

Two facts are measured and recorded, not required. ``same_bits_every_run``
is true only when every run of a kernel gave the same bits in both whole
buffers, so a fixed order showed no change between runs. For a step that
declares the order of an earlier kernel, the case records whether its
result kept that kernel's bits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy

from .gpu_reduce_bound import (
    ExactSums,
    ReductionCheck,
    case_values,
    exact_sums,
    partial_count,
    summation_depth,
)
from .gpu_reduce_config import CorrectnessControls, ReduceCase
from .gpu_reduce_plan import PLAN
from .gpu_reduce_runtime import (
    INPUT_GUARD,
    OUTPUT_SLOTS,
    PARTIAL_GUARD,
    ReductionBuffers,
    checked_launches,
)
from .gpu_reduce_source import STEP_KERNELS, GpuReduceError
from .kernel_metal_benchmark import RunBudget
from .kernel_metal_geometry import CheckedLaunch
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, array_sha256

_SENTINEL = numpy.uint32(FLOAT32_SENTINEL_BITS)


def _bits(array: numpy.ndarray) -> numpy.ndarray:
    return array.view(numpy.uint32)


def require_guards(buffers: object, name: str, groups: int, label: str) -> float:
    """Check both guarded buffers of one run; return the finite result."""

    if not isinstance(buffers, ReductionBuffers):
        raise TypeError("a run must return ReductionBuffers")
    partials, output = buffers.partials, buffers.output
    for array, size, what in (
        (partials, groups + PARTIAL_GUARD, "partials"),
        (output, OUTPUT_SLOTS, "output"),
    ):
        if (
            not isinstance(array, numpy.ndarray)
            or array.dtype != numpy.float32
            or array.shape != (size,)
        ):
            raise GpuReduceError(
                f"{label}: {name} {what} must be {size} float32 values"
            )
    written = partials[:groups]
    if not numpy.isfinite(written).all() or (_bits(written) == _SENTINEL).any():
        raise GpuReduceError(f"{label}: {name} left a partial slot unwritten")
    if not (_bits(partials[groups:]) == _SENTINEL).all():
        raise GpuReduceError(f"{label}: {name} wrote past its {groups} partials")
    if not (_bits(output[1:]) == _SENTINEL).all():
        raise GpuReduceError(f"{label}: {name} wrote past its one output")
    result = float(output[0])
    if not math.isfinite(result):
        raise GpuReduceError(
            f"{label}: {name} returned {result!r}; a read past n adds a NaN "
            "sentinel of the input tail"
        )
    return result


@dataclass(frozen=True)
class KernelOutcome:
    """One kernel's guarded runs of one case and its check against the sum."""

    kernel: str
    launch: CheckedLaunch
    runs: int
    same_bits_every_run: bool
    check: ReductionCheck

    @property
    def partials(self) -> int:
        """Pass-1 groups, one partial each."""

        return self.launch.geometry.groups[0]

    def to_mapping(self) -> dict[str, object]:
        """Return the outcome as a report records it."""

        return {
            "kernel": self.kernel,
            "launch": self.launch.to_mapping(),
            "partials": self.partials,
            "partials_guard": PARTIAL_GUARD,
            "output_guard": OUTPUT_SLOTS - 1,
            "runs": self.runs,
            "same_bits_every_run": self.same_bits_every_run,
            **self.check.to_mapping(),
        }


@dataclass(frozen=True)
class CaseResult:
    """Every kernel's outcome on one seeded case."""

    index: int
    case: ReduceCase
    input_sha256: str
    sums: ExactSums
    outcomes: tuple[KernelOutcome, ...]

    def same_order(self) -> list[dict[str, object]]:
        """For each step that keeps an earlier order, whether it kept the bits."""

        outputs = {outcome.kernel: outcome.check.output for outcome in self.outcomes}
        return [
            {
                "kernel": step.kernel,
                "same_order_as": step.same_order_as,
                "same_bits": numpy.float32(outputs[step.kernel]).tobytes()
                == numpy.float32(outputs[step.same_order_as]).tobytes(),
            }
            for step in PLAN
            if step.same_order_as is not None
        ]

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report records it."""

        return {
            "index": self.index,
            "pattern": self.case.pattern,
            "n": self.case.n,
            "input_sha256": self.input_sha256,
            "input_guard": INPUT_GUARD,
            **self.sums.to_mapping(),
            "outputs": [outcome.to_mapping() for outcome in self.outcomes],
            "same_order": self.same_order(),
        }


def check_case(
    backend: object, *, seed: int, index: int, case: ReduceCase, runs: int
) -> CaseResult:
    """Run every step kernel ``runs`` times on one case and check each result."""

    if not isinstance(case, ReduceCase):
        raise TypeError("case must be a ReduceCase")
    values = case_values(seed, index, case.pattern, case.n)
    sums = exact_sums(values)
    label = f"case {index} ({case.pattern}, n={case.n})"
    outcomes = []
    for name in STEP_KERNELS:
        launch, _ = checked_launches(backend.limits, name, case.n)
        groups = partial_count(name, case.n)
        first: ReductionBuffers | None = None
        same = True
        for _ in range(runs):
            buffers = backend.run(name, values)
            require_guards(buffers, name, groups, label)
            if first is None:
                first = buffers
                continue
            same = (
                same
                and numpy.array_equal(_bits(first.partials), _bits(buffers.partials))
                and numpy.array_equal(_bits(first.output), _bits(buffers.output))
            )
        check = ReductionCheck(
            float(first.output[0]), summation_depth(name, case.n), sums
        )
        if not check.within:
            raise GpuReduceError(
                f"{label}: {name} sits {float(check.error)!r} from the exact sum; "
                f"the bound of its order is {float(check.bound)!r}"
                + (" and the sum is exact" if sums.exact_in_any_order else "")
            )
        outcomes.append(KernelOutcome(name, launch, runs, same, check))
    return CaseResult(index, case, array_sha256(values), sums, tuple(outcomes))


def check_cases(
    backend: object, *, seed: int, controls: CorrectnessControls, budget: RunBudget
) -> tuple[CaseResult, ...]:
    """Check every configured case in order, checking the budget after each."""

    if not isinstance(controls, CorrectnessControls):
        raise TypeError("controls must be CorrectnessControls")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    results = []
    for index, case in enumerate(controls.cases):
        results.append(
            check_case(backend, seed=seed, index=index, case=case, runs=controls.runs)
        )
        budget.check()
    return tuple(results)
