"""The declared Day 80 plan: one fused kernel per family, each with its hypothesis.

This module fixes the plan before any timing. Each entry names the family,
its fused kernel and the digest of that kernel's text, the unfused PyTorch
sequence on the MPS device that it replaces, the one change that it makes,
a hypothesis, and the verdicts that it predicts over the PyTorch sequence.
A report seals the digest of this plan, and a config must name that digest,
so a hypothesis cannot move after its run.

A prediction uses the verdicts of the Day 76 ratio rule: ``faster``,
``slower`` and ``no_clear_difference``. The ratio compares the fused kernel
with the PyTorch sequence at one point of one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, VERDICTS
from .memory_access_plan import canonical_json_bytes
from .metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    FAMILY_KERNELS,
    KERNELS,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
)

TORCH_RMSNORM = "torch_rmsnorm"
TORCH_SWIGLU = "torch_swiglu"
TORCH_CROSS_ENTROPY = "torch_cross_entropy"
TORCH_SIDES = {
    RMSNORM: TORCH_RMSNORM,
    SWIGLU: TORCH_SWIGLU,
    CROSS_ENTROPY: TORCH_CROSS_ENTROPY,
}
TORCH_SEQUENCES = {
    RMSNORM: (
        "ms = x.pow(2).mean(-1, keepdim=True); inv = torch.rsqrt(ms + eps); "
        "torch.mul(x * inv, w, out=y), on the MPS device"
    ),
    SWIGLU: "torch.mul(torch.nn.functional.silu(a), b, out=y), on the MPS device",
    CROSS_ENTROPY: (
        "torch.nn.functional.cross_entropy(logits, targets, reduction='none') "
        "with int64 targets, on the MPS device"
    ),
}
MAX_TEXT = 600


@dataclass(frozen=True)
class LigerStep:
    """One family: its fused kernel, change, hypothesis and predicted verdicts."""

    family: str
    change: str
    hypothesis: str
    predicted: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise MetalLigerError(f"{self.family!r} is not a kernel family")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise MetalLigerError(
                    f"{self.family} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if (
            type(self.predicted) is not tuple
            or not self.predicted
            or len(set(self.predicted)) != len(self.predicted)
            or not set(self.predicted) <= set(VERDICTS)
        ):
            raise MetalLigerError(
                f"{self.family} must predict distinct verdicts from "
                f"{', '.join(VERDICTS)}"
            )

    @property
    def kernel(self) -> str:
        """The fused kernel of the family."""

        return FAMILY_KERNELS[self.family]

    @property
    def base(self) -> str:
        """The PyTorch side that the fused kernel replaces."""

        return TORCH_SIDES[self.family]

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report seals it, with its kernel digest."""

        return {
            "family": self.family,
            "kernel": self.kernel,
            "source_sha256": KERNELS[self.kernel].source_sha256,
            "base": self.base,
            "base_sequence": TORCH_SEQUENCES[self.family],
            "change": self.change,
            "hypothesis": self.hypothesis,
            "predicted": list(self.predicted),
        }


PLAN = (
    LigerStep(
        family=RMSNORM,
        change=(
            "Fuse the square, the mean, the rsqrt and both products into one "
            "kernel. Each thread keeps its slots of the row in registers, so "
            "the kernel reads x once and writes y once."
        ),
        hypothesis=(
            "The PyTorch sequence runs five or more kernels, and the model "
            "gives it about 40 bytes per element. The fused kernel moves about "
            "12 bytes per element. The fused kernel is faster than the "
            "PyTorch sequence by 1.1 or more at every point."
        ),
        predicted=(FASTER,),
    ),
    LigerStep(
        family=SWIGLU,
        change=(
            "Compute silu(a) * b in one pass. The value silu(a) stays in a "
            "register and never goes to device memory."
        ),
        hypothesis=(
            "The PyTorch sequence writes silu(a) and reads it again, about 20 "
            "bytes per element in the model, and the fused pass moves 12. The "
            "fused kernel is faster than the PyTorch sequence by 1.1 or more "
            "at every point."
        ),
        predicted=(FASTER,),
    ),
    LigerStep(
        family=CROSS_ENTROPY,
        change=(
            "Compute the loss of each row in one online pass over the "
            "vocabulary, with the Day 79 rescaling update. Write one float for "
            "each row and never write the softmax or the log-probabilities."
        ),
        hypothesis=(
            "If PyTorch writes the log-probabilities, it moves at least 8 bytes "
            "per logit, and the fused kernel moves 4. The online pass calls "
            "precise::exp once for each logit, and the loads probably set most "
            "of the time. The fused kernel is faster than the PyTorch call by "
            "1.1 or more, or it shows no clear difference."
        ),
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
)


def require_plan(steps: object) -> tuple[LigerStep, ...]:
    """Check the plan as a whole: every family once, in registry order."""

    if type(steps) is not tuple or not steps:
        raise MetalLigerError("a plan is a tuple of steps")
    if not all(isinstance(step, LigerStep) for step in steps):
        raise TypeError("a plan holds LigerStep values")
    if tuple(step.family for step in steps) != FAMILIES:
        raise MetalLigerError("the plan must hold every family once, in order")
    return steps


def plan_mapping(steps: tuple[LigerStep, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it."""

    require_plan(steps)
    return {"steps": [step.to_mapping() for step in steps]}


def plan_sha256(steps: tuple[LigerStep, ...] = PLAN) -> str:
    """Bind the whole plan, every hypothesis and kernel digest, to one digest."""

    return sha256(canonical_json_bytes(plan_mapping(steps))).hexdigest()


def step_of(family: str) -> LigerStep:
    """Return the plan step of one family."""

    for step in PLAN:
        if step.family == family:
            return step
    raise MetalLigerError(f"{family!r} belongs to no plan step")


require_plan(PLAN)
