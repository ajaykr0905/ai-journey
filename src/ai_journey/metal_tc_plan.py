"""The declared Day 82 plan: nine comparisons, each with its hypothesis.

This module fixes the plan before any timing of the lab. A timing point
runs seven sides on square operands of one size: the four kernels S1, M1,
M2 and M3, and ``torch.matmul`` on the MPS device in float32, float16 and
bfloat16. Each step names a first side and a base side, the one change
between them, a hypothesis and the verdicts that it predicts. A report
seals the digest of this plan with the digest of every kernel text, and a
config must name that digest, so a hypothesis cannot move after its run.

A scratch prototype ran before this plan, and the evidence folder of the
run keeps its log. It tried 64 x 64 and 32 x 32 group tiles at 1024 and
2048. With 64 x 64 tiles, M1 reached only about 0.7 TFLOP/s, so the plan
uses 32 x 32 tiles. The prototype is not a report, and the hypotheses below
cite it as a reason only.

A prediction uses the verdicts of the Day 76 ratio rule: ``faster``,
``slower`` and ``no_clear_difference``, for the first side over its base
at one point of one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes
from .metal_tc_source import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    KERNEL_NAME,
    KERNELS,
    M1,
    M2,
    M3,
    PRECISION,
    S1,
    VARIANTS,
    MetalTcError,
    library_sha256,
)

TORCH_FLOAT32 = "torch_float32"
TORCH_FLOAT16 = "torch_float16"
TORCH_BFLOAT16 = "torch_bfloat16"
TORCH_SIDES = (TORCH_FLOAT32, TORCH_FLOAT16, TORCH_BFLOAT16)
SIDES = (*VARIANTS, *TORCH_SIDES)
SIDE_PRECISION = {
    **PRECISION,
    TORCH_FLOAT32: FLOAT32,
    TORCH_FLOAT16: FLOAT16,
    TORCH_BFLOAT16: BFLOAT16,
}
TORCH_CALL = (
    "torch.matmul(a, b) on m x k and k x n tensors of the side's dtype on the "
    "MPS device; the output has the dtype of the inputs"
)
MAX_TEXT = 700


@dataclass(frozen=True)
class TcStep:
    """One comparison: a first side over a base side, with its hypothesis."""

    step: str
    first: str
    base: str
    change: str
    hypothesis: str
    predicted: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or not self.step.isascii():
            raise MetalTcError("a step needs an ASCII name")
        if self.first not in SIDES or self.base not in SIDES or self.first == self.base:
            raise MetalTcError(f"{self.step} must compare two distinct sides")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise MetalTcError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if (
            type(self.predicted) is not tuple
            or not self.predicted
            or len(set(self.predicted)) != len(self.predicted)
            or not set(self.predicted) <= set(VERDICTS)
        ):
            raise MetalTcError(
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
    TcStep(
        step="m1_over_s1",
        first=M1,
        base=S1,
        change=(
            "Use the simdgroup matrix path for float32 inputs. M1 and S1 stage "
            "the same tiles and hold 8 float accumulators per thread, but M1 "
            "multiplies 8 x 8 fragments with simdgroup_multiply_accumulate in "
            "place of scalar fma."
        ),
        hypothesis=(
            "In the scratch prototype, an MMA chain with its operands in "
            "registers ran at about 4.6 TFLOP/s, near the Day 74 FMA ceiling, so "
            "the matrix path adds no arithmetic rate. M1 needs fewer loads from "
            "threadgroup memory for each fma. In the prototype, M1 took about 0.65 "
            "to 0.75 of the time of S1. M1 is faster than S1 by 1.1 or more at "
            "every size."
        ),
        predicted=(FASTER,),
    ),
    TcStep(
        step="m2_over_m1",
        first=M2,
        base=M1,
        change=(
            "Read half inputs in place of float32 inputs on the matrix path. The "
            "tiles and fragments hold half the bytes, and the accumulators stay "
            "float32."
        ),
        hypothesis=(
            "The MMA chain of the prototype ran at the same rate for half and "
            "float, so the arithmetic rate does not change. Half the bytes can "
            "help only where loads limit M1. In the prototype, M2 was faster than "
            "M1 at 1024 and a little slower at 2048. M2 is faster than M1 or shows "
            "no clear difference."
        ),
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
    TcStep(
        step="m3_over_m1",
        first=M3,
        base=M1,
        change=(
            "Read bfloat inputs in place of float32 inputs on the matrix path. "
            "Bfloat has the bytes of half and the exponent range of float32."
        ),
        hypothesis=(
            "Bfloat moves the same bytes as half, and the MMA chain of the "
            "prototype ran at the same rate for bfloat and half. M3 has the speed "
            "of M2: faster than M1 or no clear difference."
        ),
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
    TcStep(
        step="m3_over_s1",
        first=M3,
        base=S1,
        change=(
            "Compare bfloat inputs on the matrix path with float32 inputs without "
            "it: the pair of the hands-on task, fp32 without the matrix path and "
            "bf16 with it."
        ),
        hypothesis=(
            "This step adds the two changes before it. In the prototype, M3 took "
            "about 0.55 to 0.8 of the time of S1. M3 is faster than S1 by 1.1 or "
            "more at every size."
        ),
        predicted=(FASTER,),
    ),
    TcStep(
        step="torch_float16_over_torch_float32",
        first=TORCH_FLOAT16,
        base=TORCH_FLOAT32,
        change="Call torch.matmul on float16 tensors in place of float32 tensors.",
        hypothesis=(
            "In the prototype at 2048, the float16 call was about 1.1 times as "
            "fast as the float32 call. The float16 call is faster or shows no "
            "clear difference."
        ),
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
    TcStep(
        step="torch_bfloat16_over_torch_float32",
        first=TORCH_BFLOAT16,
        base=TORCH_FLOAT32,
        change="Call torch.matmul on bfloat16 tensors in place of float32 tensors.",
        hypothesis=(
            "In the prototype at 2048, the bfloat16 call had about the speed of "
            "the float16 call. The bfloat16 call is faster or shows no clear "
            "difference."
        ),
        predicted=(FASTER, NO_CLEAR_DIFFERENCE),
    ),
    TcStep(
        step="m1_over_torch_float32",
        first=M1,
        base=TORCH_FLOAT32,
        change="Compare M1 with the float32 PyTorch call on the same tensors.",
        hypothesis=(
            "In the prototype at 2048, the float32 call ran at about 3.8 to 4.0 "
            "TFLOP/s and M1 at about 1.8. M1 is slower than the PyTorch call at "
            "every size."
        ),
        predicted=(SLOWER,),
    ),
    TcStep(
        step="m2_over_torch_float16",
        first=M2,
        base=TORCH_FLOAT16,
        change="Compare M2 with the float16 PyTorch call on the same tensors.",
        hypothesis=(
            "The PyTorch call writes float16 outputs and M2 writes float32, but "
            "both do the same 2 n**3 operations. M2 is slower than the PyTorch "
            "call at every size."
        ),
        predicted=(SLOWER,),
    ),
    TcStep(
        step="m3_over_torch_bfloat16",
        first=M3,
        base=TORCH_BFLOAT16,
        change="Compare M3 with the bfloat16 PyTorch call on the same tensors.",
        hypothesis="M3 is slower than the bfloat16 PyTorch call at every size.",
        predicted=(SLOWER,),
    ),
)


def require_plan(steps: object) -> tuple[TcStep, ...]:
    """Check the plan as a whole: steps with distinct names."""

    if type(steps) is not tuple or not steps:
        raise MetalTcError("a plan is a tuple of steps")
    if not all(isinstance(step, TcStep) for step in steps):
        raise TypeError("a plan holds TcStep values")
    if len({step.step for step in steps}) != len(steps):
        raise MetalTcError("each step of the plan needs its own name")
    return steps


def kernel_digests() -> dict[str, str]:
    """The digest of every kernel text that a side of the plan launches."""

    return {
        KERNEL_NAME[variant]: KERNELS[KERNEL_NAME[variant]].source_sha256
        for variant in VARIANTS
    }


def plan_mapping(steps: tuple[TcStep, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it, with the kernel and library digests."""

    require_plan(steps)
    return {
        "sides": list(SIDES),
        "precisions": {side: SIDE_PRECISION[side] for side in SIDES},
        "torch_call": TORCH_CALL,
        "kernels": kernel_digests(),
        "libraries": {name: library_sha256(name) for name in ("main", "bfloat")},
        "steps": [step.to_mapping() for step in steps],
    }


def plan_sha256(steps: tuple[TcStep, ...] = PLAN) -> str:
    """Bind the whole plan, every hypothesis and kernel digest, to one digest."""

    return sha256(canonical_json_bytes(plan_mapping(steps))).hexdigest()


require_plan(PLAN)
