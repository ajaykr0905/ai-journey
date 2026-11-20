"""The declared Day 78 scan plan: one hypothesis for each kernel, before any timing.

Each hypothesis names its kernel, the side that it compares with (its base),
the change that the kernel makes and a prediction for each pinned length:
the verdicts of the Day 76 ratio rule that it allows, ``faster``, ``slower``
or ``no_clear_difference``. The plan also binds the digest of every kernel
text, so a report that seals the plan digest seals the kernels it predicted.
A config must name this digest, so no run can test other hypotheses.

The plan entered the repository before the first timed run. Before it, two
untimed scripts ran scans on the development GPU and read no clock: one ran
a prototype text once at each length, and one ran the library text three
times on each input. The plan uses one fact of the second script:
``torch.cumsum`` gave other bits from run to run at 65537 and 1048577
elements, and the three scans did not. The scan guide describes both
scripts.

The pinned lengths are 2^16, 2^20 and 2^24 and one ragged length that leaves
a partial block at every level. A point at another length has no prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .gpu_scan_source import (
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    add_mapping,
    library_sha256,
    require_length,
)
from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER, VERDICTS
from .memory_access_plan import canonical_json_bytes

VENDOR = "torch_cumsum"
COPY = "stream_copy"
SIDES = (*KERNELS, VENDOR, COPY)
BASES = (*KERNELS, VENDOR)
RAGGED_LENGTH = 5_000_003
LENGTHS = (1 << 16, 1 << 20, 1 << 24, RAGGED_LENGTH)
MAX_TEXT = 800
_STEP = re.compile(r"S[0-9]")


@dataclass(frozen=True)
class ScanHypothesis:
    """One kernel's change and its predicted verdict over its base at each length."""

    step: str
    kernel: str
    base: str
    change: str
    hypothesis: str
    predictions: tuple[tuple[int, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or _STEP.fullmatch(self.step) is None:
            raise GpuScanError("a step id must be S and one digit")
        if self.kernel not in KERNELS:
            raise GpuScanError(f"{self.kernel!r} is not a registered scan kernel")
        if self.base not in BASES or self.base == self.kernel:
            raise GpuScanError(
                f"{self.step} needs another kernel or the vendor as base"
            )
        for text, label in ((self.change, "change"), (self.hypothesis, "hypothesis")):
            if (
                not isinstance(text, str)
                or not text.isascii()
                or not 1 <= len(text) <= MAX_TEXT
            ):
                raise GpuScanError(
                    f"{self.step} {label} must be ASCII text of 1 to {MAX_TEXT} "
                    "characters"
                )
        if type(self.predictions) is not tuple:
            raise TypeError("predictions must be a tuple of (length, verdicts) pairs")
        lengths = []
        for entry in self.predictions:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("a prediction is a (length, verdicts) pair")
            length, verdicts = entry
            lengths.append(require_length(length))
            if (
                type(verdicts) is not tuple
                or not verdicts
                or len(set(verdicts)) != len(verdicts)
                or not set(verdicts) <= set(VERDICTS)
            ):
                raise GpuScanError(
                    f"{self.step} must predict distinct verdicts from "
                    f"{', '.join(VERDICTS)}"
                )
        if tuple(lengths) != LENGTHS:
            raise GpuScanError(
                f"{self.step} must predict each pinned length once, in order"
            )

    def predicted(self, length: int) -> tuple[str, ...] | None:
        """The verdicts allowed at one length; ``None`` for a length with none."""

        return dict(self.predictions).get(length)

    def to_mapping(self) -> dict[str, object]:
        """Return the hypothesis as a report seals it."""

        return {
            "step": self.step,
            "kernel": self.kernel,
            "base": self.base,
            "change": self.change,
            "hypothesis": self.hypothesis,
            "predictions": [
                {"length": length, "verdicts": list(verdicts)}
                for length, verdicts in self.predictions
            ],
        }


def _same(*verdicts: str) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple((length, verdicts) for length in LENGTHS)


PLAN = (
    ScanHypothesis(
        step="S0",
        kernel=S0_HILLIS_STEELE,
        base=VENDOR,
        change=(
            "Scan each block of 1024 elements with the 10 Hillis-Steele steps in "
            "two threadgroup buffers, with one barrier at each step. Then scan the "
            "block sums with the same kernel and add the sum of the earlier blocks "
            "to each element."
        ),
        hypothesis=(
            "PyTorch does not report which kernel torch.cumsum runs. An untimed "
            "check before this plan ran each side three times. torch.cumsum gave "
            "other bits from run to run at 65537 and 1048577 elements, and the "
            "three scans of this lab did not. Sums in an order that changes from "
            "run to run can come from a single-pass scan. Such a scan reads and "
            "writes each element once, in one dispatch. S0 moves about 16 bytes for each "
            "element in 3 or 5 dispatches. On Day 73, the median time of torch.add "
            "was 0.85 to 1.24 times that of a Metal vector add. Thus S0 is slower "
            "than torch.cumsum at every length."
        ),
        predictions=_same(SLOWER),
    ),
    ScanHypothesis(
        step="S1",
        kernel=S1_BLELLOCH,
        base=S0_HILLIS_STEELE,
        change=(
            "Scan each block with 512 threads in one threadgroup buffer: an "
            "up-sweep and a down-sweep of 10 levels each, with a barrier at each "
            "level. Element j takes the exclusive sum of element j + 1, and the "
            "last element takes the block total."
        ),
        hypothesis=(
            "S1 makes about 2 adds for each element, and S0 makes about 9. But S1 "
            "waits at 22 barriers in each block, and S0 waits at 11. At each "
            "level, the lanes of S1 read words 2 to 512 words apart. On Day 74, "
            "threadgroup reads at strides of 2 to 32 words took 2.0 to 21.5 times "
            "as long as reads at stride 1. Both kernels make the same dispatches "
            "and move the same modeled bytes. Thus S1 is not faster than S0 at any "
            "length, and at 2^16 the run shows no clear difference."
        ),
        predictions=(
            (LENGTHS[0], (NO_CLEAR_DIFFERENCE,)),
            *((length, (NO_CLEAR_DIFFERENCE, SLOWER)) for length in LENGTHS[1:]),
        ),
    ),
    ScanHypothesis(
        step="S2",
        kernel=S2_SIMD,
        base=S0_HILLIS_STEELE,
        change=(
            "Scan each run of 32 elements with simd_prefix_inclusive_sum. SIMD "
            "group 0 scans the 32 run totals and moves them up one lane with "
            "simd_shuffle_up. Each thread adds the total of the runs before its "
            "own, with 2 barriers in each block."
        ),
        hypothesis=(
            "SIMD-group functions are the Metal counterpart of CUDA warp shuffles, "
            "not the same instruction. S2 replaces the 10 threadgroup steps and 11 "
            "barriers of S0 with two SIMD-group scans and 2 barriers. At 2^16, the "
            "dispatches probably set the time, so the run shows no clear "
            "difference. S0 makes about 31 threadgroup accesses for each element. "
            "If each access takes the Day 74 time of 1.3 ps for a stride-1 "
            "threadgroup read, S0 spends about 42 us on them at 2^20. Its 16.8 MB "
            "of modeled traffic take 168 us at the Day 74 copy ceiling of about "
            "100 GB/s. Thus S2 is faster than S0 by 1.1 or more at 2^20, 2^24 and "
            "the ragged length."
        ),
        predictions=(
            (LENGTHS[0], (NO_CLEAR_DIFFERENCE,)),
            *((length, (FASTER,)) for length in LENGTHS[1:]),
        ),
    ),
)


def require_plan(plan: object) -> tuple[ScanHypothesis, ...]:
    """Check the plan as a whole: S0, S1, ... in order, one step for each kernel."""

    if type(plan) is not tuple or not plan:
        raise GpuScanError("a plan is a tuple of hypotheses")
    if not all(isinstance(entry, ScanHypothesis) for entry in plan):
        raise TypeError("a plan holds ScanHypothesis values")
    if [entry.step for entry in plan] != [f"S{index}" for index in range(len(plan))]:
        raise GpuScanError("the steps must run S0, S1, ... in order")
    if [entry.kernel for entry in plan] != list(KERNELS):
        raise GpuScanError("the plan needs one step for each kernel, in registry order")
    return plan


def plan_mapping(plan: tuple[ScanHypothesis, ...] = PLAN) -> dict[str, object]:
    """Return the plan as a report seals it, with the digest of every kernel."""

    require_plan(plan)
    return {
        "lengths": list(LENGTHS),
        "sides": list(SIDES),
        "vendor": VENDOR,
        "copy": COPY,
        "kernels": {name: kernel.source_sha256 for name, kernel in KERNELS.items()},
        "add_kernel_sha256": add_mapping()["source_sha256"],
        "library_sha256": library_sha256(),
        "hypotheses": [entry.to_mapping() for entry in plan],
    }


def plan_sha256(plan: tuple[ScanHypothesis, ...] = PLAN) -> str:
    """Bind the whole plan, each kernel digest included, to one SHA-256 digest."""

    return sha256(canonical_json_bytes(plan_mapping(plan))).hexdigest()


def hypothesis_of(kernel: str) -> ScanHypothesis:
    """Return the step that declares ``kernel``."""

    for entry in PLAN:
        if entry.kernel == kernel:
            return entry
    raise GpuScanError(f"{kernel!r} has no declared hypothesis")


require_plan(PLAN)
