"""The declared Day 81 plan: three comparisons, each with its hypothesis.

This module fixes the plan before any timing of the lab. A timing point
runs three sides on the same device rows: N0, the naive three-kernel
attention, F1, the flash-attention kernel, and
``torch.nn.functional.scaled_dot_product_attention`` on the MPS device.
Each step of the plan names a first side and a base side, the one change
between them, a hypothesis and the verdicts that it predicts. A report
seals the digest of this plan with the digest of every kernel text, and a
config must name that digest, so a hypothesis cannot move after its run.

A prediction uses the verdicts of the Day 76 ratio rule: ``faster``,
``slower`` and ``no_clear_difference``, for the first side over its base
at one point of one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .matmul_tiled_comparison import FASTER, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes
from .metal_flash_source import F1, KERNELS, MODES, N0, MetalFlashError, kernel_names

TORCH_SDPA = "torch_sdpa"
SIDES = (N0, F1, TORCH_SDPA)
TORCH_CALL = (
    "torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal) "
    "on 1 x H x N x 64 float32 views of the device rows, with the default scale "
    "1 / sqrt(64), on the MPS device"
)
MAX_TEXT = 700


@dataclass(frozen=True)
class FlashStep:
    """One comparison: a first side over a base side, with its hypothesis."""

    step: str
    first: str
    base: str
    change: str
    hypothesis: str
    predicted: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or not self.step.isascii():
            raise MetalFlashError("a step needs an ASCII name")
        if self.first not in SIDES or self.base not in SIDES or self.first == self.base:
            raise MetalFlashError(f"{self.step} must compare two distinct sides")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise MetalFlashError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if (
            type(self.predicted) is not tuple
            or not self.predicted
            or len(set(self.predicted)) != len(self.predicted)
            or not set(self.predicted) <= set(VERDICTS)
        ):
            raise MetalFlashError(
                f"{self.step} must predict distinct verdicts from {', '.join(VERDICTS)}"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report seals it."""

        return {
            "step": self.step,
            "first": self.first,
            "base": self.base,
            "change": self.change,
            "hypothesis": self.hypothesis,
            "predicted": list(self.predicted),
        }


PLAN = (
    FlashStep(
        step="n0_over_torch",
        first=N0,
        base=TORCH_SDPA,
        change=(
            "Compare the naive three-kernel attention with the PyTorch call. N0 "
            "writes the N x N scores, reads and writes them in the softmax and "
            "reads them again for P V, and it uses no SIMD-group matrix unit."
        ),
        hypothesis=(
            "N0 moves 16 bytes for each visible pair through device memory and "
            "computes each score with scalar fma. N0 is slower than the PyTorch "
            "call at every point."
        ),
        predicted=(SLOWER,),
    ),
    FlashStep(
        step="f1_over_n0",
        first=F1,
        base=N0,
        change=(
            "Fuse the three steps into one kernel. Load K and V in tiles of 32 "
            "rows into threadgroup memory and keep a running max and sum, so S "
            "and P never go to device memory."
        ),
        hypothesis=(
            "F1 does the same arithmetic as N0 but removes the N x N traffic and "
            "two launches. F1 is faster than N0 by 1.1 or more at every point."
        ),
        predicted=(FASTER,),
    ),
    FlashStep(
        step="f1_over_torch",
        first=F1,
        base=TORCH_SDPA,
        change=(
            "Compare the flash kernel with the PyTorch call. F1 uses scalar fma "
            "for both products, and four threads share one query row."
        ),
        hypothesis=(
            "A scratch prototype of F1 ran before this plan was written, at 512, "
            "2048 and 4096 tokens, and took about 2.5 to 3 times the time of the "
            "PyTorch call. Scalar fma probably limits F1. F1 is slower than the "
            "PyTorch call at every point."
        ),
        predicted=(SLOWER,),
    ),
)


def require_plan(steps: object) -> tuple[FlashStep, ...]:
    """Check the plan as a whole: three steps with distinct names."""

    if type(steps) is not tuple or not steps:
        raise MetalFlashError("a plan is a tuple of steps")
    if not all(isinstance(step, FlashStep) for step in steps):
        raise TypeError("a plan holds FlashStep values")
    if len({step.step for step in steps}) != len(steps):
        raise MetalFlashError("each step of the plan needs its own name")
    return steps


def kernel_digests() -> dict[str, str]:
    """The digest of every kernel text that a side of the plan launches."""

    return {
        name: KERNELS[name].source_sha256
        for variant in (N0, F1)
        for mode in MODES
        for name in kernel_names(variant, mode)
    }


def plan_mapping(steps: tuple[FlashStep, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it, with the kernel digests."""

    require_plan(steps)
    return {
        "sides": list(SIDES),
        "torch_call": TORCH_CALL,
        "kernels": kernel_digests(),
        "steps": [step.to_mapping() for step in steps],
    }


def plan_sha256(steps: tuple[FlashStep, ...] = PLAN) -> str:
    """Bind the whole plan, every hypothesis and kernel digest, to one digest."""

    return sha256(canonical_json_bytes(plan_mapping(steps))).hexdigest()


require_plan(PLAN)
