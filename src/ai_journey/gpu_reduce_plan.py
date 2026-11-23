"""The declared reduction plan: one step per kernel, each with its hypothesis.

Curriculum Day 78 asks for a sum reduction kernel, then a faster one with
warp shuffles, and a benchmark of both. This module fixes the plan before
any timing: each step names its kernel and the digest of its text, the one
change that it makes, a hypothesis and the verdicts that it predicts over
its base for each size class. A report seals the digest of this plan, so a
hypothesis cannot move after its run.

The steps form one chain. Each step's base is the step before it, so its
ratio to that base is the speedup of its own change, and its ratio to R0 is
the speedup of every change so far. R0 has no step before it, so its base is
``torch.sum`` on the MPS device, the vendor side.

A prediction uses the verdicts of the Day 76 ratio rule (``faster``,
``slower`` and ``no_clear_difference``) for each of three size classes,
whose bounds are part of the plan:

- ``small``: fewer than 2^18 elements, 1 MiB. The host dispatch of each call
  probably sets the time.
- ``medium``: 2^18 to 2^22 - 1 elements. The input can stay in a cache
  between the calls of a sample.
- ``large``: 2^22 elements, 16 MiB, or more.

The class names state a hypothesis about each range, not a measured fact.
A step can also declare that its kernel adds the same pairs in the same
order as an earlier kernel; the run then records whether every result
kept that kernel's bits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .gpu_reduce_source import (
    KERNELS,
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    STEP_KERNELS,
    GpuReduceError,
)
from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes

TORCH_SUM = "torch_sum"
SMALL = "small"
MEDIUM = "medium"
LARGE = "large"
MAX_ELEMENTS = 1 << 26
# (class, first n, last n): the classes cover 1 to MAX_ELEMENTS with no gap.
SIZE_CLASSES = (
    (SMALL, 1, (1 << 18) - 1),
    (MEDIUM, 1 << 18, (1 << 22) - 1),
    (LARGE, 1 << 22, MAX_ELEMENTS),
)
CLASS_NAMES = tuple(name for name, _, _ in SIZE_CLASSES)
MAX_TEXT = 600
_STEP = re.compile(r"R[0-9]")


def size_class(n: int) -> str:
    """Name the class of a size by the declared bounds."""

    for name, first, last in SIZE_CLASSES:
        if type(n) is int and first <= n <= last:
            return name
    raise GpuReduceError(f"{n!r} is outside 1 to {MAX_ELEMENTS} elements")


@dataclass(frozen=True)
class ReduceStep:
    """One step: its kernel, the change, the hypothesis and its predictions."""

    step: str
    kernel: str
    change: str
    hypothesis: str
    base: str
    predictions: tuple[tuple[str, tuple[str, ...]], ...]
    same_order_as: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or _STEP.fullmatch(self.step) is None:
            raise GpuReduceError("a step id must be R and one digit")
        if self.kernel not in STEP_KERNELS or KERNELS[self.kernel].step != self.step:
            raise GpuReduceError(f"{self.step} must name its own registered kernel")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise GpuReduceError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if self.base != TORCH_SUM and self.base not in STEP_KERNELS:
            raise GpuReduceError(f"{self.step} base {self.base!r} is unknown")
        if self.base == self.kernel:
            raise GpuReduceError(f"{self.step} cannot be its own base")
        if (
            type(self.predictions) is not tuple
            or tuple(name for name, _ in self.predictions) != CLASS_NAMES
        ):
            raise GpuReduceError(
                f"{self.step} must predict each size class once: "
                + ", ".join(CLASS_NAMES)
            )
        for _, verdicts in self.predictions:
            if (
                type(verdicts) is not tuple
                or not verdicts
                or len(set(verdicts)) != len(verdicts)
                or not set(verdicts) <= set(VERDICTS)
            ):
                raise GpuReduceError(
                    f"{self.step} must predict distinct verdicts from "
                    f"{', '.join(VERDICTS)}"
                )
        if self.same_order_as is not None and (
            self.same_order_as not in STEP_KERNELS
            or STEP_KERNELS.index(self.same_order_as) >= STEP_KERNELS.index(self.kernel)
        ):
            raise GpuReduceError(
                f"{self.step} can keep the order of an earlier kernel only"
            )

    def predicted(self, n: int) -> tuple[str, ...]:
        """The verdicts that the hypothesis allows at one size."""

        return dict(self.predictions)[size_class(n)]

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report seals it, with its kernel's digest."""

        return {
            "step": self.step,
            "kernel": self.kernel,
            "source_sha256": KERNELS[self.kernel].source_sha256,
            "change": self.change,
            "hypothesis": self.hypothesis,
            "base": self.base,
            "predictions": [
                {"size_class": name, "verdicts": list(verdicts)}
                for name, verdicts in self.predictions
            ],
            "same_order_as": self.same_order_as,
        }


def _predict(small, medium, large) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return ((SMALL, small), (MEDIUM, medium), (LARGE, large))


PLAN = (
    ReduceStep(
        step="R0",
        kernel=R0,
        change=(
            "Load one element per thread. Then add pairs in threadgroup memory "
            "with interleaved addressing. At stride s, only the threads with "
            "t % (2 s) == 0 add, and s doubles from 1 to 128."
        ),
        hypothesis=(
            "R0 is slower than torch.sum from 2^18 elements. At each level, the "
            "modulo test keeps most lanes of every SIMD group idle. Every SIMD "
            "group still takes the branch, and a group waits at 9 barriers for "
            "each 256 elements. Below 2^18 elements, the host dispatch of a call "
            "probably sets the time. R0 dispatches two short kernels, so it is "
            "faster than torch.sum or shows no clear difference."
        ),
        base=TORCH_SUM,
        predictions=_predict((FASTER, NO_CLEAR_DIFFERENCE), (SLOWER,), (SLOWER,)),
    ),
    ReduceStep(
        step="R1",
        kernel=R1,
        change=(
            "Add pairs with sequential addressing: at stride s, the threads "
            "t < s add shared[t + s], and s halves from 128 to 1."
        ),
        hypothesis=(
            "R1 is faster than R0 by 1.1 or more from 2^18 elements. While s is "
            "32 or more, a whole SIMD group takes the branch or skips it. Thus "
            "only the last 5 levels diverge, and no thread calculates a modulo. "
            "Below 2^18 elements, the dispatch probably sets the time, so R1 "
            "shows no clear difference from R0 or is faster."
        ),
        base=R0,
        predictions=_predict((NO_CLEAR_DIFFERENCE, FASTER), (FASTER,), (FASTER,)),
    ),
    ReduceStep(
        step="R2",
        kernel=R2,
        change=(
            "Give each thread two elements 256 apart, and add them during the "
            "global load. One group then covers 512 elements, and the grid has "
            "half the groups."
        ),
        hypothesis=(
            "R2 is faster than R1 by 1.1 or more from 2^18 elements. Each thread "
            "has two loads in flight before its first barrier, so more loads are "
            "in flight. Also, the tree work for each element halves. Below 2^18 "
            "elements, R2 shows no clear difference from R1."
        ),
        base=R1,
        predictions=_predict((NO_CLEAR_DIFFERENCE,), (FASTER,), (FASTER,)),
    ),
    ReduceStep(
        step="R3",
        kernel=R3,
        change=(
            "Stop the threadgroup tree at 32 values. Then add them in SIMD group "
            "0 with simd_shuffle_down at deltas 16, 8, 4, 2 and 1, with no "
            "barrier."
        ),
        hypothesis=(
            "R3 removes 5 of the 9 barriers of a group. From 2^22 elements, the "
            "device loads probably set the time of R2, so R3 shows no clear "
            "difference from R2. From 2^18 to 2^22 elements, a cache can serve "
            "the input, so the barriers can matter, and R3 is faster or shows no "
            "clear difference. Below 2^18 elements, R3 shows no clear "
            "difference. R3 adds the same pairs in the same order as R2, so each "
            "result keeps the bits of R2."
        ),
        base=R2,
        predictions=_predict(
            (NO_CLEAR_DIFFERENCE,),
            (NO_CLEAR_DIFFERENCE, FASTER),
            (NO_CLEAR_DIFFERENCE,),
        ),
        same_order_as=R2,
    ),
    ReduceStep(
        step="R4",
        kernel=R4,
        change=(
            "Add the two loaded values, and add each SIMD group with simd_sum. "
            "Write the 8 group sums to threadgroup memory, and wait at one "
            "barrier. Then SIMD group 0 adds the 8 sums with simd_sum."
        ),
        hypothesis=(
            "R4 waits at 1 barrier for each group instead of 4. From 2^22 "
            "elements, the loads probably set the time, so R4 shows no clear "
            "difference from R3. From 2^18 to 2^22 elements, R4 is faster or "
            "shows no clear difference. Below 2^18 elements, R4 shows no clear "
            "difference. simd_sum does not state its order of additions. Thus "
            "the bound of R4 allows any order in a SIMD group, and its bits can "
            "differ from R3."
        ),
        base=R3,
        predictions=_predict(
            (NO_CLEAR_DIFFERENCE,),
            (NO_CLEAR_DIFFERENCE, FASTER),
            (NO_CLEAR_DIFFERENCE,),
        ),
    ),
    ReduceStep(
        step="R5",
        kernel=R5,
        change=(
            "Launch a fixed grid of 1024 groups. Each thread adds the elements "
            "gid, gid + 262144, gid + 2 * 262144 and so on in one loop. Then it "
            "reduces as R4 does. The combine pass thus always adds 1024 partials."
        ),
        hypothesis=(
            "R5 shows no clear difference from R4 from 2^18 elements. Both read "
            "each element once, and the combine pass of R4 reads at most 131072 "
            "partials, 1/512 of the input. Below 2^18 elements, most R5 threads "
            "have no element, so R5 is slower or shows no clear difference. Each "
            "R5 thread adds up to n / 262144 elements in one chain, so the bound "
            "of R5 grows with n."
        ),
        base=R4,
        predictions=_predict(
            (NO_CLEAR_DIFFERENCE, SLOWER),
            (NO_CLEAR_DIFFERENCE,),
            (NO_CLEAR_DIFFERENCE,),
        ),
    ),
)


def require_plan(steps: object) -> tuple[ReduceStep, ...]:
    """Check the plan as a whole: one chain over every step kernel, in order.

    The steps run R0 to R5. R0 starts from ``torch.sum``, and each later step
    starts from the kernel of the step before it.
    """

    if type(steps) is not tuple or not all(
        isinstance(step, ReduceStep) for step in steps
    ):
        raise GpuReduceError("a plan is a tuple of ReduceStep values")
    if [step.step for step in steps] != [f"R{index}" for index in range(6)]:
        raise GpuReduceError("the steps must run R0 to R5 in order")
    if [step.kernel for step in steps] != list(STEP_KERNELS):
        raise GpuReduceError("each step must hold its own kernel, in order")
    if steps[0].base != TORCH_SUM:
        raise GpuReduceError("R0 must start from torch.sum")
    for before, after in zip(steps, steps[1:]):
        if after.base != before.kernel:
            raise GpuReduceError(
                f"{after.step} must start from {before.kernel}, the kernel of "
                f"{before.step}"
            )
    return steps


def plan_mapping(steps: tuple[ReduceStep, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it: the classes, the base and the steps."""

    require_plan(steps)
    return {
        "vendor": TORCH_SUM,
        "reference_kernel": R0,
        "size_classes": [
            {"name": name, "first_n": first, "last_n": last}
            for name, first, last in SIZE_CLASSES
        ],
        "steps": [step.to_mapping() for step in steps],
    }


def plan_sha256(steps: tuple[ReduceStep, ...] = PLAN) -> str:
    """Bind the whole plan, each hypothesis and kernel digest, to one digest."""

    return sha256(canonical_json_bytes(plan_mapping(steps))).hexdigest()


def step_of(kernel: str) -> ReduceStep:
    """Return the step whose kernel is ``kernel``."""

    for step in PLAN:
        if step.kernel == kernel:
            return step
    raise GpuReduceError(f"{kernel!r} belongs to no reduction step")


require_plan(PLAN)
