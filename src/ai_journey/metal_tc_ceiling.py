"""Join the Day 74 compute and bandwidth ceilings to a Day 82 run.

A matmul of size n does ``2 n**3`` operations on ``(2 e + c) n**2`` bytes,
so its intensity grows with n, and the compute ceiling limits it long
before the bandwidth ceiling does. This lab reads both ceilings from one
Day 74 gpu-ceilings report through the Day 81 loader: the best median
GFLOP/s of the Day 74 FMA chains and the better median GB/s of its two
stream kernels. The Day 74 verifier must accept the report first, so a
tampered report sets no ceiling.

The roofline of a side at one point is ``min(F, I B)``: the FMA ceiling
``F`` or the intensity ``I`` times the bandwidth ceiling ``B``, whichever is
lower. A fraction of a ceiling is a measurement only when the run and the
ceiling report are both measurements from the same interpreter, libraries
and platform. Neither ceiling is a vendor peak or a hardware limit, so a
fraction above 1 is possible.
"""

from __future__ import annotations

from pathlib import Path

from .metal_flash_ceiling import FlashCeiling, load_flash_ceiling
from .metal_flash_source import MetalFlashError
from .metal_tc_source import MetalTcError

ROOFLINE_RULE = (
    "roofline GFLOP/s = min(F, I B), with F the FMA ceiling in GFLOP/s, I the "
    "compulsory intensity of the side in FLOP/byte and B the bandwidth ceiling "
    "in GB/s; a side is compute-bound by this model when I B >= F"
)


def roofline(intensity: float, bandwidth: float, fma: float) -> float:
    """Return ``min(F, I B)`` in GFLOP/s."""

    for value, label in (
        (intensity, "intensity"),
        (bandwidth, "bandwidth"),
        (fma, "fma"),
    ):
        if type(value) is not float or not value > 0:
            raise MetalTcError(f"the {label} of a roofline must be a positive float")
    return min(fma, intensity * bandwidth)


def ridge_point(ceiling: FlashCeiling) -> float:
    """The intensity in FLOP/byte where both ceilings meet."""

    return ceiling.fma_gflops_per_second / ceiling.bandwidth.gb_per_second


def load_tc_ceiling(path: Path) -> FlashCeiling:
    """Read one gpu-ceilings report through the verified Day 74 loader."""

    try:
        return load_flash_ceiling(path)
    except MetalTcError:
        raise
    except MetalFlashError as error:
        raise MetalTcError(str(error)) from error
