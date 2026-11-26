"""An explained speedup model of fp32, tf32 and bf16 matmuls on A100 and H100.

This is a model, not a measurement. It takes only the declared inputs of the
config, and it states its rule in ``RULE``:

1. A square N x N x N matmul does N^3 multiply-adds, 2 N^3 FLOPs.
2. One SM does a declared number of dense FMAs per clock on each math path.
   Per clock, bf16 on the Tensor Cores does 16 times the FMAs of FP32 on the
   FMA lanes, and tf32 8 times, on A100 and on H100.
3. The compute ceiling of a path is its listed dense peak: the listed figure,
   or half of it when the datasheet counts 2:4 sparsity.
4. The kernel reads A and B once and writes the float32 C once:
   N^2 x (2 x input bytes + 4) bytes. No kernel moves fewer bytes, so the
   model gives the shortest memory time. A real kernel moves more.
5. The time is the larger of FLOPs / compute ceiling and bytes / bandwidth,
   the roofline of ``matmul_model_roofline``. A size is memory-bound on a
   path when its memory time is the larger one.

``ratio`` divides the time of each path by the bf16 time. Where all paths
are compute-bound, the ratio is the ratio of the peaks. Where bf16 is
memory-bound, its time stops shrinking with its peak, and the ratio falls.
The model ignores launch overhead, the tile count (N = 512 gives 16
programs of 128 x 128 for 108 or 132 SMs), caches, and the clock under load.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from .matmul_model_reference import MatmulModelError
from .matmul_model_roofline import Ceilings, roofline
from .matmul_model_traffic import ratio_mapping
from .triton_tc_config import PATHS, Device, TcModelConfig
from .triton_tc_kernel import TritonTcError

INPUT_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2}
RATIO_PATHS = ("fp32", "tf32", "bf16")
LABEL = "declared_model_not_a_measurement"
RULE = (
    "flops = 2 * N**3 (one FMA is 2 FLOPs)",
    "compute ceiling = listed dense peak; a figure with sparsity is halved",
    "bytes = N**2 * (2 * input_bytes + 4): A and B read once, float32 C once",
    "time = max(flops / compute ceiling, bytes / bandwidth)",
    "memory-bound when bytes / bandwidth > flops / compute ceiling",
    "ratio = time(path) / time(bf16)",
)
OMITTED = (
    "launch overhead",
    "tile count and SM load: 128 x 128 tiles give 16 programs at N = 512",
    "L2 and shared memory reuse beyond one read of A and B",
    "the clock under load and thermal limits",
)


def _gflops_text(value: Fraction) -> str:
    """Write a GFLOP/s value as the canonical decimal that ``Ceilings`` takes."""

    for places in range(7):
        scaled = value * 10**places
        if scaled.denominator == 1:
            whole, part = divmod(scaled.numerator, 10**places)
            return f"{whole}.{part:0{places}d}" if places else str(whole)
    raise TritonTcError(f"{value} GFLOP/s has more than 6 decimals")


def ceilings(device: Device, path: str) -> Ceilings:
    """Return the compute and bandwidth ceilings of one path on one device."""

    if type(device) is not Device:
        raise TypeError("device must be Device")
    try:
        return Ceilings(
            label=f"{device.key}-{path}",
            compute_gflops=_gflops_text(device.path(path).dense_tflops * 1000),
            bandwidth_gbps=str(device.memory_bandwidth.value),
        )
    except MatmulModelError as error:
        raise TritonTcError(str(error)) from error


def traffic_bytes(size: int, path: str, output_bytes: int = 4) -> int:
    """Return the bytes of one read of A and B and one write of C."""

    if path not in INPUT_BYTES:
        raise TritonTcError(f"path must be one of {PATHS}")
    if type(size) is not int or size < 1:
        raise TritonTcError("size must be a positive integer")
    return size * size * (2 * INPUT_BYTES[path] + output_bytes)


def point(device: Device, size: int, path: str) -> dict[str, object]:
    """Return the times in microseconds and the bound of one path at one size."""

    flops = 2 * size**3
    moved = traffic_bytes(size, path)
    limits = ceilings(device, path)
    place = roofline(Fraction(flops, moved), limits)
    compute_us = Fraction(flops) / (limits.compute * 1000)
    memory_us = Fraction(moved) / (limits.bandwidth * 1000)
    return {
        "path": path,
        "flops": flops,
        "bytes": moved,
        "intensity": Fraction(flops, moved),
        "ridge": limits.ridge_point,
        "compute_us": compute_us,
        "memory_us": memory_us,
        "time_us": max(compute_us, memory_us),
        "bound": place.bound,
    }


def size_row(device: Device, size: int) -> dict[str, object]:
    """Return every path at one size, the ratios and the memory-bound paths."""

    points = {path: point(device, size, path) for path in PATHS}
    base = points["bf16"]["time_us"]
    return {
        "size": size,
        "points": points,
        "ratio": {path: points[path]["time_us"] / base for path in PATHS},
        "memory_bound": [path for path in PATHS if points[path]["bound"] == "memory"],
    }


def device_summary(device: Device) -> dict[str, object]:
    """Return the per-clock and peak ratios, implied clocks and ridge points."""

    rates = {path: device.path(path).rate for path in PATHS}
    peaks = {path: device.path(path).dense_tflops for path in PATHS}
    return {
        "key": device.key,
        "name": device.name,
        "target": device.target,
        "sm_count": device.sms,
        "bandwidth_gbps": device.bandwidth_gbps,
        "fma_per_sm_per_clock": rates,
        "dense_tflops": peaks,
        "per_clock_ratio": {
            path: Fraction(rates["bf16"], rates[path]) for path in PATHS
        },
        "peak_ratio": {path: peaks["bf16"] / peaks[path] for path in PATHS},
        "implied_clock_mhz": {path: device.implied_clock_mhz(path) for path in PATHS},
        "reproduction": {path: device.reproduction(path) for path in PATHS},
        "ridge": {path: ceilings(device, path).ridge_point for path in PATHS},
    }


def run_model(config: TcModelConfig) -> dict[str, object]:
    """Return the model of every declared device at every declared size."""

    if type(config) is not TcModelConfig:
        raise TypeError("config must be TcModelConfig")
    return {
        "label": LABEL,
        "rule": list(RULE),
        "omitted": list(OMITTED),
        "sizes": list(config.sizes),
        "devices": [
            {
                **device_summary(device),
                "rows": [size_row(device, size) for size in config.sizes],
            }
            for device in config.devices
        ],
    }


def _exact(value: object) -> object:
    if type(value) is Fraction:
        return ratio_mapping(value)
    if isinstance(value, Mapping):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact(item) for item in value]
    return value


def model_mapping(config: TcModelConfig) -> dict[str, object]:
    """Return the model as JSON-safe data: each fraction exact and rounded."""

    return _exact(run_model(config))


def ratio_text(ratio: Mapping[str, Fraction], places: int = 2) -> str:
    """Write ``fp32 : tf32 : bf16`` with the bf16 time as 1."""

    return " : ".join(
        f"{float(ratio[path]):.{places}f}" if path != "bf16" else "1"
        for path in RATIO_PATHS
    )
