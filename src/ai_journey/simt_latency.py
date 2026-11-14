"""Little's-law latency hiding from declared bandwidth and latency inputs.

Little's law says concurrency = throughput x latency. To sustain a declared
DRAM bandwidth ``B`` bytes/s at a declared latency of ``L`` SM clock cycles on
``S`` SMs clocked at ``f`` Hz, each SM must keep ``B * L / (S * f)`` bytes in
flight. The same law sizes arithmetic pipelines: accepting ``R``
warp-instructions per cycle across a dependent-issue latency of ``L`` cycles
needs ``R * L`` independent instructions in flight. Latencies here are always
declared inputs; nothing is measured, and cache hits, coalescing losses, DRAM
efficiency, and instruction mix are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .simt_device import SimtArchitecture, SimtDevice, SimtError, require_int

# CUDA C++ Programming Guide 5.3.2: global memory instructions read or write
# words of 1, 2, 4, 8, or 16 bytes per thread.
ACCESS_WIDTHS_BYTES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class MemoryAccessPattern:
    """Bytes one warp keeps outstanding: width x active lanes x independent loads."""

    bytes_per_lane: int
    active_lanes: int
    independent_loads: int

    def __post_init__(self) -> None:
        if (
            type(self.bytes_per_lane) is not int
            or self.bytes_per_lane not in ACCESS_WIDTHS_BYTES
        ):
            raise SimtError("bytes_per_lane must be a 1, 2, 4, 8, or 16 byte access")
        require_int(self.active_lanes, "active_lanes", minimum=1, maximum=64)
        require_int(self.independent_loads, "independent_loads", minimum=1, maximum=64)

    @property
    def bytes_per_warp(self) -> int:
        return self.bytes_per_lane * self.active_lanes * self.independent_loads

    def to_mapping(self) -> dict[str, int]:
        return {
            "bytes_per_lane": self.bytes_per_lane,
            "active_lanes": self.active_lanes,
            "independent_loads": self.independent_loads,
        }


@dataclass(frozen=True)
class MemoryLatencyHiding:
    """Bytes and warps in flight needed to sustain one device's DRAM bandwidth."""

    device_key: str
    latency_cycles: int
    pattern: MemoryAccessPattern
    bytes_per_cycle_per_sm: Fraction
    bytes_in_flight_per_sm: Fraction
    latency_seconds: Fraction
    warps_needed: int
    max_warps_per_sm: int
    resident_warps: int
    sustainable_bandwidth_fraction: Fraction
    dram_bandwidth_bytes_per_second: int

    @property
    def full_bandwidth_within_warp_limit(self) -> bool:
        return self.warps_needed <= self.max_warps_per_sm

    @property
    def sustainable_bandwidth_bytes_per_second(self) -> Fraction:
        return (
            self.sustainable_bandwidth_fraction * self.dram_bandwidth_bytes_per_second
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "device": self.device_key,
            "latency_cycles": self.latency_cycles,
            "latency_seconds": float(self.latency_seconds),
            "access": self.pattern.to_mapping(),
            "bytes_per_warp": self.pattern.bytes_per_warp,
            "bytes_per_cycle_per_sm": float(self.bytes_per_cycle_per_sm),
            "bytes_in_flight_per_sm": float(self.bytes_in_flight_per_sm),
            "warps_needed": self.warps_needed,
            "max_warps_per_sm": self.max_warps_per_sm,
            "full_bandwidth_within_warp_limit": self.full_bandwidth_within_warp_limit,
            "resident_warps": self.resident_warps,
            "sustainable_bandwidth_fraction": float(
                self.sustainable_bandwidth_fraction
            ),
            "sustainable_bandwidth_bytes_per_second": float(
                self.sustainable_bandwidth_bytes_per_second
            ),
        }


def memory_latency_hiding(
    device: SimtDevice,
    *,
    latency_cycles: int,
    pattern: MemoryAccessPattern,
    resident_warps: int,
) -> MemoryLatencyHiding:
    """Size the memory parallelism one SM needs at a declared DRAM latency.

    The bandwidth share each SM must sustain is the device bandwidth divided by
    its SM count, and the latency is converted with the declared boost clock.
    The sustainable fraction is ``resident_warps * bytes_per_warp`` over the
    bytes that must be in flight, capped at one.
    """

    if not isinstance(device, SimtDevice):
        raise TypeError("device must be SimtDevice")
    if not isinstance(pattern, MemoryAccessPattern):
        raise TypeError("pattern must be MemoryAccessPattern")
    architecture = device.architecture
    latency = require_int(latency_cycles, "latency_cycles", minimum=1, maximum=100_000)
    if pattern.active_lanes > architecture.warp_size:
        raise SimtError("active_lanes cannot exceed the warp size")
    warps = require_int(
        resident_warps,
        "resident_warps",
        minimum=1,
        maximum=architecture.max_warps_per_sm,
    )
    per_cycle = Fraction(
        device.dram_bandwidth_bytes_per_second,
        device.sm_count * device.boost_clock_hz,
    )
    in_flight = per_cycle * latency
    return MemoryLatencyHiding(
        device_key=device.key,
        latency_cycles=latency,
        pattern=pattern,
        bytes_per_cycle_per_sm=per_cycle,
        bytes_in_flight_per_sm=in_flight,
        latency_seconds=Fraction(latency, device.boost_clock_hz),
        warps_needed=math.ceil(in_flight / pattern.bytes_per_warp),
        max_warps_per_sm=architecture.max_warps_per_sm,
        resident_warps=warps,
        sustainable_bandwidth_fraction=min(
            Fraction(1), warps * pattern.bytes_per_warp / in_flight
        ),
        dram_bandwidth_bytes_per_second=device.dram_bandwidth_bytes_per_second,
    )


ARITHMETIC_PIPELINES = ("fp32", "issue")


@dataclass(frozen=True)
class ArithmeticLatencyHiding:
    """Independent warp-instructions an SM needs to hide a dependent latency."""

    compute_capability: str
    pipeline: str
    warp_instructions_per_cycle: Fraction
    latency_cycles: int
    independent_instructions: int
    instructions_in_flight: Fraction
    warps_needed: int
    max_warps_per_sm: int
    resident_warps: int
    sustainable_throughput_fraction: Fraction

    @property
    def full_throughput_within_warp_limit(self) -> bool:
        return self.warps_needed <= self.max_warps_per_sm

    def to_mapping(self) -> dict[str, object]:
        return {
            "compute_capability": self.compute_capability,
            "pipeline": self.pipeline,
            "warp_instructions_per_cycle": float(self.warp_instructions_per_cycle),
            "latency_cycles": self.latency_cycles,
            "independent_instructions": self.independent_instructions,
            "instructions_in_flight": float(self.instructions_in_flight),
            "warps_needed": self.warps_needed,
            "max_warps_per_sm": self.max_warps_per_sm,
            "full_throughput_within_warp_limit": (
                self.full_throughput_within_warp_limit
            ),
            "resident_warps": self.resident_warps,
            "sustainable_throughput_fraction": float(
                self.sustainable_throughput_fraction
            ),
        }


def arithmetic_latency_hiding(
    architecture: SimtArchitecture,
    *,
    pipeline: str,
    latency_cycles: int,
    independent_instructions: int,
    resident_warps: int,
) -> ArithmeticLatencyHiding:
    """Apply Little's law to an arithmetic pipeline with a declared latency.

    ``fp32`` accepts ``fp32_lanes_per_sm / warp_size`` warp-instructions per
    cycle per SM. ``issue`` is the guide's maximum-throughput case of one
    instruction per warp scheduler per cycle, where hiding ``L`` cycles needs
    ``4L`` instructions on a four-scheduler SM. Each resident warp contributes
    ``independent_instructions`` instructions that do not wait on each other.
    """

    if not isinstance(architecture, SimtArchitecture):
        raise TypeError("architecture must be SimtArchitecture")
    if pipeline == "fp32":
        rate = Fraction(architecture.fp32_lanes_per_sm, architecture.warp_size)
    elif pipeline == "issue":
        rate = Fraction(architecture.warp_schedulers_per_sm)
    else:
        raise SimtError(f"pipeline must be one of {ARITHMETIC_PIPELINES}")
    latency = require_int(latency_cycles, "latency_cycles", minimum=1, maximum=10_000)
    parallelism = require_int(
        independent_instructions, "independent_instructions", minimum=1, maximum=64
    )
    warps = require_int(
        resident_warps,
        "resident_warps",
        minimum=1,
        maximum=architecture.max_warps_per_sm,
    )
    in_flight = rate * latency
    return ArithmeticLatencyHiding(
        compute_capability=architecture.compute_capability,
        pipeline=pipeline,
        warp_instructions_per_cycle=rate,
        latency_cycles=latency,
        independent_instructions=parallelism,
        instructions_in_flight=in_flight,
        warps_needed=math.ceil(in_flight / parallelism),
        max_warps_per_sm=architecture.max_warps_per_sm,
        resident_warps=warps,
        sustainable_throughput_fraction=min(
            Fraction(1), warps * parallelism / in_flight
        ),
    )
