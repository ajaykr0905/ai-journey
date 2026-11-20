"""The declared performance checklist: ordered steps, each with its hypothesis.

Curriculum Day 77 asks for the checklist, an application of every item to
the matmul, and the speedup after each change. This module fixes the plan
before any timing: each step names the checklist item that it applies, the
one change that it makes, a hypothesis with the verdicts that it predicts,
the kernel that it starts from and the kernels that it adds. A report seals
the digest of this plan, so a hypothesis cannot move after its run.

The steps form one chain. Each step starts from the kernel that the step
before it keeps (``chain``), so its ratio to its base is the speedup of its
own change, and its ratio to S0 is the speedup of every change so far. S6
adds two group sizes beside the S5 kernel and keeps the S5 kernel in the
chain. S0 and S1 change nothing: the Day 74 and Day 76 labs applied them,
and the run times their kernels again as evidence.

The items follow the Day 77 tracker notes: coalescing, occupancy, tiling,
arithmetic intensity and memory vs compute bound. The lab adds control
divergence for the ragged shape. A prediction uses the verdicts of the
Day 76 ratio rule: ``faster``, ``slower`` and ``no_clear_difference``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_variants import (
    DAY_77,
    S2_COARSE,
    S3_BLOCK,
    S4_VECTOR,
    S5_PAD,
    S6_LARGE,
    S6_SMALL,
    S7_INTERIOR,
    TILED_32,
    VARIANTS,
    require_known,
)
from .matmul_metal_source import MATMUL_X_COLUMNS, MATMUL_X_ROWS
from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes

COALESCING = "coalescing"
OCCUPANCY = "occupancy"
TILING = "tiling"
ARITHMETIC_INTENSITY = "arithmetic_intensity"
MEMORY_VS_COMPUTE = "memory_vs_compute_bound"
CONTROL_DIVERGENCE = "control_divergence"
ITEMS = (
    COALESCING,
    OCCUPANCY,
    TILING,
    ARITHMETIC_INTENSITY,
    MEMORY_VS_COMPUTE,
    CONTROL_DIVERGENCE,
)
ITEM_SOURCE = (
    "the Day 77 tracker notes list coalescing, occupancy, tiling, arithmetic "
    "intensity and memory vs compute bound; this lab adds control divergence "
    "for the ragged shape"
)
ALREADY_APPLIED = "already_applied"
NEW_KERNEL = "new_kernel"
STATUSES = (ALREADY_APPLIED, NEW_KERNEL)
REFERENCE = MATMUL_X_COLUMNS
MAX_TEXT = 600
_STEP = re.compile(r"S[0-9]")


@dataclass(frozen=True)
class ChecklistStep:
    """One step: the item, the change, the hypothesis and the kernels it times."""

    step: str
    item: str
    status: str
    change: str
    hypothesis: str
    base: str
    predictions: tuple[tuple[str, tuple[str, ...]], ...]
    chain: str

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or _STEP.fullmatch(self.step) is None:
            raise MatmulChecklistError("a step id must be S and one digit")
        if self.item not in ITEMS:
            raise MatmulChecklistError(f"item must be one of {', '.join(ITEMS)}")
        if self.status not in STATUSES:
            raise MatmulChecklistError(f"status must be one of {', '.join(STATUSES)}")
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise MatmulChecklistError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if type(self.predictions) is not tuple or not self.predictions:
            raise MatmulChecklistError(f"{self.step} must predict at least one kernel")
        for entry in self.predictions:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("a prediction is a (variant, verdicts) pair")
            name, verdicts = entry
            if name not in VARIANTS or name == self.base:
                raise MatmulChecklistError(
                    f"{self.step} predicts {name!r}, which is not a new kernel of "
                    "this step"
                )
            if (
                type(verdicts) is not tuple
                or not verdicts
                or len(set(verdicts)) != len(verdicts)
                or not set(verdicts) <= set(VERDICTS)
            ):
                raise MatmulChecklistError(
                    f"{self.step} must predict distinct verdicts from "
                    f"{', '.join(VERDICTS)}"
                )
        if self.base not in VARIANTS:
            raise MatmulChecklistError(f"{self.step} base {self.base!r} is unknown")
        if self.chain not in (self.base, *self.variants):
            raise MatmulChecklistError(f"{self.step} must keep one of its kernels")
        new = all(VARIANTS[name].origin == DAY_77 for name in self.variants)
        if new != (self.status == NEW_KERNEL):
            raise MatmulChecklistError(
                f"{self.step} adds new kernels exactly when its status is "
                f"{NEW_KERNEL}"
            )

    @property
    def variants(self) -> tuple[str, ...]:
        """The kernels this step times beside its base, in prediction order."""

        return tuple(name for name, _ in self.predictions)

    def predicted(self, variant: str) -> tuple[str, ...]:
        """The verdicts that the hypothesis allows for one of the step's kernels."""

        return dict(self.predictions)[variant]

    def to_mapping(self) -> dict[str, object]:
        """Return the step as a report seals it."""

        return {
            "step": self.step,
            "item": self.item,
            "status": self.status,
            "change": self.change,
            "hypothesis": self.hypothesis,
            "base": self.base,
            "predictions": [
                {"variant": name, "verdicts": list(verdicts)}
                for name, verdicts in self.predictions
            ],
            "chain": self.chain,
        }


CHECKLIST = (
    ChecklistStep(
        step="S0",
        item=COALESCING,
        status=ALREADY_APPLIED,
        change=(
            "Map x to the columns of C, so that the lanes of a SIMD group read "
            "consecutive elements of B (Day 74)."
        ),
        hypothesis=(
            "matmul_x_columns is faster than matmul_x_rows at every point. On "
            "Day 74, matmul_x_rows ran at 0.30 to 0.63 of the matmul_x_columns "
            "rate. Its lanes read elements of A that are K floats apart."
        ),
        base=MATMUL_X_ROWS,
        predictions=((MATMUL_X_COLUMNS, (FASTER,)),),
        chain=MATMUL_X_COLUMNS,
    ),
    ChecklistStep(
        step="S1",
        item=TILING,
        status=ALREADY_APPLIED,
        change=(
            "Stage 32 x 32 tiles of A and B in threadgroup memory, with one "
            "output for each thread (Day 76)."
        ),
        hypothesis=(
            "Tiling alone does not make the kernel faster. On Day 76, tile 32 "
            "ran at 0.87 to 0.96 of the naive rate. The barriers and the "
            "threadgroup reads probably cost more time than the device loads "
            "that the tiles save."
        ),
        base=MATMUL_X_COLUMNS,
        predictions=((TILED_32, (SLOWER, NO_CLEAR_DIFFERENCE)),),
        chain=TILED_32,
    ),
    ChecklistStep(
        step="S2",
        item=ARITHMETIC_INTENSITY,
        status=NEW_KERNEL,
        change=(
            "Give each thread 4 outputs in one column of the 32 x 32 tile, with "
            "4 register accumulators, in groups of 32 x 8 threads."
        ),
        hypothesis=(
            "Each B value from threadgroup memory now serves 4 fma steps, so "
            "threadgroup reads fall from 2 to 1.25 for each fma. Each barrier "
            "also covers 4 times the work of a thread. S2 is faster than S1 by "
            "1.1 or more."
        ),
        base=TILED_32,
        predictions=((S2_COARSE, (FASTER,)),),
        chain=S2_COARSE,
    ),
    ChecklistStep(
        step="S3",
        item=ARITHMETIC_INTENSITY,
        status=NEW_KERNEL,
        change=(
            "Give each thread a 4 x 4 block of outputs from 64 x 64 tiles, with "
            "K phases of 16."
        ),
        hypothesis=(
            "Each value from threadgroup memory now serves 4 fma steps, so "
            "threadgroup reads fall from 1.25 to 0.5 for each fma. The model "
            "also halves the device loads for each output. S3 is faster than S2 "
            "by 1.1 or more."
        ),
        base=S2_COARSE,
        predictions=((S3_BLOCK, (FASTER,)),),
        chain=S3_BLOCK,
    ),
    ChecklistStep(
        step="S4",
        item=MEMORY_VS_COMPUTE,
        status=NEW_KERNEL,
        change=(
            "Stage A and B with float4 loads and stores when K and N are "
            "multiples of 4, and with scalars otherwise."
        ),
        hypothesis=(
            "The no-cache model gives S3 15.9 FLOPs for each byte at 4096 cubed. "
            "That is below the ridge point of 38.8 from the Day 76 gap run, so "
            "the model calls S3 memory bound. On Day 76, the naive kernel ran 12 "
            "times above its no-cache roof, so caches probably serve most "
            "repeated loads. Each thread of S3 also stages only 8 values for the "
            "256 fma steps of a phase. S4 shows no clear difference from S3."
        ),
        base=S3_BLOCK,
        predictions=((S4_VECTOR, (NO_CLEAR_DIFFERENCE,)),),
        chain=S4_VECTOR,
    ),
    ChecklistStep(
        step="S5",
        item=TILING,
        status=NEW_KERNEL,
        change=(
            "Add 4 floats of padding to each tile row in threadgroup memory, as "
            "a control for bank conflicts."
        ),
        hypothesis=(
            "At each k step, the lanes of a SIMD group of S4 read two A words "
            "and 16 consecutive B words. The 32-bank model of the Day 74 "
            "memory-access lab finds no conflict there, so padding cannot "
            "remove one. Padding moves the B rows off their 64-byte boundaries, "
            "and the gpu-ceilings lab measured slower consecutive reads off "
            "128-byte boundaries. S5 is not faster than S4."
        ),
        base=S4_VECTOR,
        predictions=((S5_PAD, (NO_CLEAR_DIFFERENCE, SLOWER)),),
        chain=S5_PAD,
    ),
    ChecklistStep(
        step="S6",
        item=OCCUPANCY,
        status=NEW_KERNEL,
        change=(
            "Run the S5 kernel in 8 x 8 and 32 x 32 groups, with tiles of "
            "32 x 32 and 128 x 128."
        ),
        hypothesis=(
            "No occupancy counter exists here, so this step records the time "
            "and the pipeline limits only. The 8 x 8 group doubles the modeled "
            "device loads for each output and holds only 2 SIMD groups, so it "
            "is slower than S5. The 32 x 32 group halves those loads, but each "
            "group holds 1024 threads with 16 accumulators, so it is not faster "
            "than S5."
        ),
        base=S5_PAD,
        predictions=(
            (S6_SMALL, (SLOWER,)),
            (S6_LARGE, (NO_CLEAR_DIFFERENCE, SLOWER)),
        ),
        chain=S5_PAD,
    ),
    ChecklistStep(
        step="S7",
        item=CONTROL_DIVERGENCE,
        status=NEW_KERNEL,
        change=(
            "Give interior groups a path with no guards: no zero-fill selects, "
            "a constant k-loop bound and no store guards."
        ),
        hypothesis=(
            "At the ragged shape, only 78 of the 1504 groups lie at an edge, so "
            "divergence costs little there. In the interior path, the constant "
            "loop bound lets the compiler unroll the k loop. S7 is faster than "
            "S5 by 1.1 or more."
        ),
        base=S5_PAD,
        predictions=((S7_INTERIOR, (FASTER,)),),
        chain=S7_INTERIOR,
    ),
)


def require_checklist(steps: object) -> tuple[ChecklistStep, ...]:
    """Check the plan as a whole: one chain that covers every registered kernel.

    The steps run S0, S1, ... in order. The first step starts from
    ``matmul_x_rows`` and keeps ``REFERENCE``; each later step starts from
    the kernel that the step before it keeps. Every registered kernel is a
    base or a new kernel of exactly one step, and every item has a step.
    """

    if type(steps) is not tuple or not steps:
        raise MatmulChecklistError("a checklist is a tuple of steps")
    if not all(isinstance(step, ChecklistStep) for step in steps):
        raise TypeError("a checklist holds ChecklistStep values")
    if [step.step for step in steps] != [f"S{index}" for index in range(len(steps))]:
        raise MatmulChecklistError("the steps must run S0, S1, ... in order")
    first = steps[0]
    if first.base != MATMUL_X_ROWS or first.chain != REFERENCE:
        raise MatmulChecklistError(
            f"S0 must start from {MATMUL_X_ROWS} and keep {REFERENCE}"
        )
    for before, after in zip(steps, steps[1:]):
        if after.base != before.chain:
            raise MatmulChecklistError(
                f"{after.step} must start from {before.chain}, the kernel that "
                f"{before.step} keeps"
            )
    timed = require_known(
        [first.base, *(name for step in steps for name in step.variants)]
    )
    if set(timed) != set(VARIANTS):
        raise MatmulChecklistError("every registered kernel must have a step")
    missing = set(ITEMS) - {step.item for step in steps}
    if missing:
        raise MatmulChecklistError(
            f"the checklist has no step for {', '.join(sorted(missing))}"
        )
    return steps


def checklist_mapping(steps: tuple[ChecklistStep, ...] = CHECKLIST) -> dict:
    """Return the plan as a report seals it: the items, their source and the steps."""

    require_checklist(steps)
    return {
        "items": list(ITEMS),
        "item_source": ITEM_SOURCE,
        "reference": REFERENCE,
        "steps": [step.to_mapping() for step in steps],
    }


def checklist_sha256(steps: tuple[ChecklistStep, ...] = CHECKLIST) -> str:
    """Bind the whole plan, every hypothesis included, to one SHA-256 digest."""

    return sha256(canonical_json_bytes(checklist_mapping(steps))).hexdigest()


def step_of(variant: str) -> ChecklistStep:
    """Return the step that adds ``variant``; S0 also owns its base."""

    for step in CHECKLIST:
        if variant in step.variants or (step.step == "S0" and variant == step.base):
            return step
    raise MatmulChecklistError(f"{variant!r} belongs to no checklist step")


require_checklist(CHECKLIST)
