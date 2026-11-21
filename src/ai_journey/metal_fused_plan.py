"""The declared fusion plan: one step per softmax kernel, each with its hypothesis.

This module fixes the plan before any timing. Each step names its variant
and the digest of its kernel text, the one change that it makes, a
hypothesis, and the verdicts that it predicts over its base. A report seals
the digest of this plan, and a config must name that digest, so a
hypothesis cannot move after its run.

The steps form one chain, U0, F1, F2, F3. Each later step's base is the step
before it, so its ratio to that base is the speedup of its own change, and
its ratio to U0 is the speedup of every change so far. U0 has no step before
it, so its base is ``torch.softmax(x, dim=-1)`` on the MPS device, the
vendor side. A prediction uses the verdicts of the Day 76 ratio rule:
``faster``, ``slower`` and ``no_clear_difference``.

A step can also declare that its kernel does the same arithmetic in the same
order as an earlier step. The run then measures whether every output kept
the bits of that step; the declaration predicts, and the run records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes
from .metal_fused_source import (
    F1,
    F2,
    F3,
    STEP_VARIANTS,
    U0,
    VARIANTS,
    MetalFusedError,
)

TORCH_SOFTMAX = "torch_softmax"
BASELINE = U0
MAX_TEXT = 600
_STEP = re.compile(r"[UF][0-9]")


@dataclass(frozen=True)
class FusedStep:
    """One step: its variant, change, hypothesis, base and predicted verdicts."""

    step: str
    variant: str
    change: str
    hypothesis: str
    base: str
    predicted: tuple[str, ...]
    same_bits_as: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or _STEP.fullmatch(self.step) is None:
            raise MetalFusedError("a step id must be U or F and one digit")
        if self.variant not in STEP_VARIANTS or VARIANTS[self.variant].step != (
            self.step
        ):
            raise MetalFusedError(f"{self.step} must name its own registered variant")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise MetalFusedError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if self.base != TORCH_SOFTMAX and self.base not in STEP_VARIANTS:
            raise MetalFusedError(f"{self.step} base {self.base!r} is unknown")
        if self.base == self.variant:
            raise MetalFusedError(f"{self.step} cannot be its own base")
        if (
            type(self.predicted) is not tuple
            or not self.predicted
            or len(set(self.predicted)) != len(self.predicted)
            or not set(self.predicted) <= set(VERDICTS)
        ):
            raise MetalFusedError(
                f"{self.step} must predict distinct verdicts from {', '.join(VERDICTS)}"
            )
        if self.same_bits_as is not None and (
            self.same_bits_as not in STEP_VARIANTS
            or STEP_VARIANTS.index(self.same_bits_as)
            >= STEP_VARIANTS.index(self.variant)
        ):
            raise MetalFusedError(
                f"{self.step} can keep the bits of an earlier step only"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report seals it, with its kernel digest."""

        return {
            "step": self.step,
            "variant": self.variant,
            "source_sha256": VARIANTS[self.variant].source_sha256,
            "change": self.change,
            "hypothesis": self.hypothesis,
            "base": self.base,
            "predicted": list(self.predicted),
            "same_bits_as": self.same_bits_as,
        }


PLAN = (
    FusedStep(
        step="U0",
        variant=U0,
        change=(
            "Compute the softmax in three kernels: the max of each row, the sum "
            "of exp(x - max) and the normalize pass. The max and the sum go "
            "through two buffers in device memory."
        ),
        hypothesis=(
            "U0 reads each row three times and writes it once, so the model "
            "gives it about twice the bytes of one read and one write. It also "
            "launches three kernels for each call. U0 is slower than "
            "torch.softmax at every point."
        ),
        base=TORCH_SOFTMAX,
        predicted=(SLOWER,),
    ),
    FusedStep(
        step="F1",
        variant=F1,
        change=(
            "Fuse the three passes into one kernel. Each thread keeps its 16 "
            "slots of the row in registers, and two trees in threadgroup memory "
            "give the max and the sum."
        ),
        hypothesis=(
            "F1 reads each element once and writes it once, so the model gives "
            "it about half the bytes of U0, with one launch in place of three. "
            "F1 is faster than U0 by 1.1 or more at every point. F1 adds the "
            "same values in the same order as U0, so its outputs can keep the "
            "U0 bits."
        ),
        base=U0,
        predicted=(FASTER,),
        same_bits_as=U0,
    ),
    FusedStep(
        step="F2",
        variant=F2,
        change=(
            "Replace each threadgroup tree with simd_max or simd_sum in each "
            "SIMD group and one threadgroup step."
        ),
        hypothesis=(
            "A tree over G threads needs log2(G) + 1 barriers for each value, "
            "and F2 needs one. F2 moves the same modeled bytes as F1, and the "
            "loads of the row probably set most of the time. F2 is faster than "
            "F1 or shows no clear difference."
        ),
        base=F1,
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
    FusedStep(
        step="F3",
        variant=F3,
        change=(
            "Keep a running max and a running sum in one pass over the slots, "
            "rescale the sum when the max grows, and merge the (max, sum) pairs "
            "in one threadgroup step."
        ),
        hypothesis=(
            "F3 moves the same modeled bytes as F2 and needs one barrier in "
            "place of two. It calls precise::exp twice for each element, and F2 "
            "calls it once. F3 is not faster than F2: it shows no clear "
            "difference or it is slower."
        ),
        base=F2,
        predicted=(NO_CLEAR_DIFFERENCE, SLOWER),
    ),
)


def require_plan(steps: object) -> tuple[FusedStep, ...]:
    """Check the plan as a whole: one chain from U0 over every registered step.

    The steps run U0, F1, F2, F3 in registry order. U0 starts from
    ``torch.softmax`` and each later step starts from the step before it.
    """

    if type(steps) is not tuple or not steps:
        raise MetalFusedError("a plan is a tuple of steps")
    if not all(isinstance(step, FusedStep) for step in steps):
        raise TypeError("a plan holds FusedStep values")
    if tuple(step.variant for step in steps) != STEP_VARIANTS:
        raise MetalFusedError("the plan must hold every step once, in order")
    if steps[0].base != TORCH_SOFTMAX:
        raise MetalFusedError(f"U0 must start from {TORCH_SOFTMAX}")
    for before, after in zip(steps, steps[1:]):
        if after.base != before.variant:
            raise MetalFusedError(
                f"{after.step} must start from {before.variant}, the step before it"
            )
    return steps


def plan_mapping(steps: tuple[FusedStep, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it."""

    require_plan(steps)
    return {
        "baseline": BASELINE,
        "vendor": TORCH_SOFTMAX,
        "steps": [step.to_mapping() for step in steps],
    }


def plan_sha256(steps: tuple[FusedStep, ...] = PLAN) -> str:
    """Bind the whole plan, every hypothesis and kernel digest, to one digest."""

    return sha256(canonical_json_bytes(plan_mapping(steps))).hexdigest()


def step_of(variant: str) -> FusedStep:
    """Return the plan step of one variant."""

    for step in PLAN:
        if step.variant == variant:
            return step
    raise MetalFusedError(f"{variant!r} belongs to no plan step")


require_plan(PLAN)
