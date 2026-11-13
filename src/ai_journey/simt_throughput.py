"""Peak FP32 throughput and roofline balance of declared SIMT devices.

The device ceiling reuses :mod:`ai_journey.roofline`: peak FP32 is SMs x FP32
lanes x 2 FLOPs per fused multiply-add x boost clock, the memory ceiling is the
declared DRAM bandwidth, and the ridge point is their ratio. Workload counts
are compulsory-traffic conventions; nothing is executed or measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .resource_accounting import bytes_per_element
from .roofline import (
    KernelWorkload,
    RooflineEstimate,
    RooflineHardware,
    analyze_roofline,
)
from .simt_device import SimtDevice, require_int


def _require_device(device: object) -> SimtDevice:
    if not isinstance(device, SimtDevice):
        raise TypeError("device must be SimtDevice")
    return device


def device_roofline(device: SimtDevice) -> RooflineHardware:
    """Return the declared FP32 and DRAM ceilings as roofline hardware."""

    checked = _require_device(device)
    return RooflineHardware(
        peak_flops_per_second=float(checked.peak_fp32_flops_per_second),
        memory_bandwidth_bytes_per_second=float(
            checked.dram_bandwidth_bytes_per_second
        ),
    )


@dataclass(frozen=True)
class DeviceThroughput:
    """Exact FP32 peak, DRAM bandwidth, and ridge point of one device."""

    device_key: str
    peak_fp32_flops_per_second: int
    dram_bandwidth_bytes_per_second: int

    @property
    def ridge_point_flops_per_byte(self) -> Fraction:
        return Fraction(
            self.peak_fp32_flops_per_second, self.dram_bandwidth_bytes_per_second
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "device": self.device_key,
            "peak_fp32_flops_per_second": self.peak_fp32_flops_per_second,
            "dram_bandwidth_bytes_per_second": self.dram_bandwidth_bytes_per_second,
            "ridge_point_flops_per_byte": float(self.ridge_point_flops_per_byte),
        }


def device_throughput(device: SimtDevice) -> DeviceThroughput:
    """Return the declared ceilings that bound any kernel on ``device``."""

    checked = _require_device(device)
    return DeviceThroughput(
        device_key=checked.key,
        peak_fp32_flops_per_second=checked.peak_fp32_flops_per_second,
        dram_bandwidth_bytes_per_second=checked.dram_bandwidth_bytes_per_second,
    )


def vector_add_workload(*, elements: int, dtype: str) -> KernelWorkload:
    """Count ``c[i] = a[i] + b[i]``: one add, two reads, and one write each."""

    count = require_int(elements, "elements", minimum=1, maximum=2**62)
    width = bytes_per_element(dtype)
    return KernelWorkload(
        kind="vector_add",
        flops=count,
        bytes_read=2 * count * width,
        bytes_written=count * width,
    )


def rgb_to_grayscale_workload(
    *,
    width: int,
    height: int,
    input_bytes_per_pixel: int,
    output_bytes_per_pixel: int,
    flops_per_pixel: int,
) -> KernelWorkload:
    """Count a weighted RGB-to-luma conversion that touches each pixel once."""

    columns = require_int(width, "width", minimum=1, maximum=2**31 - 1)
    rows = require_int(height, "height", minimum=1, maximum=2**31 - 1)
    read = require_int(
        input_bytes_per_pixel, "input_bytes_per_pixel", minimum=1, maximum=64
    )
    written = require_int(
        output_bytes_per_pixel, "output_bytes_per_pixel", minimum=1, maximum=64
    )
    flops = require_int(flops_per_pixel, "flops_per_pixel", minimum=1, maximum=1024)
    pixels = columns * rows
    return KernelWorkload(
        kind="rgb_to_grayscale",
        flops=pixels * flops,
        bytes_read=pixels * read,
        bytes_written=pixels * written,
    )


def roofline_on_device(
    workload: KernelWorkload, device: SimtDevice
) -> RooflineEstimate:
    """Classify one workload against the device's declared ceilings."""

    return analyze_roofline(workload, device_roofline(device))
