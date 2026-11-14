"""Independent verification of sealed SIMT execution-model reports.

Verification trusts no derived number. It checks the schema, the fixed
evidence boundary, the exclusions, and the SHA-256 seal; rebuilds the declared
plan, whose hardware must still equal the pinned source-backed catalog; then
re-derives every derived field with this module's own integer and fraction
arithmetic (occupancy-calculator rules, coverage counts, Little's law, waves,
FP32 peaks, roofline bounds) and names the first field that differs. Finally
the report must also equal a fresh build by the analysis modules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction

from .resource_accounting import DTYPE_BYTES
from .simt_device import SimtArchitecture, SimtDevice, SimtError
from .simt_json import canonical_json_bytes
from .simt_plan import SimtPlan
from .simt_report import (
    SIMT_EVIDENCE_BOUNDARY,
    SIMT_EXCLUSIONS,
    SIMT_REPORT_KIND,
    build_simt_report,
    seal_simt_report,
    simt_report_digest,
)


class SimtVerificationError(SimtError):
    """Raised when a SIMT report cannot be reproduced from its declared inputs."""


_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "declared_inputs",
    "devices",
    "vector_add",
    "rgb_to_grayscale",
    "occupancy_sweep",
    "latency_hiding",
    "wave_quantization",
    "summary",
    "evidence_boundary",
    "exclusions",
    "integrity",
}
_LIMITS = ("warps", "registers", "shared_memory", "blocks")


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _compare(expected: object, actual: object, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise SimtVerificationError(f"{path} fields do not match recomputation")
        for key in sorted(expected):
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise SimtVerificationError(f"{path} rows do not match recomputation")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise SimtVerificationError(f"{path} does not match independent recomputation")


def _datasheet(device: SimtDevice) -> dict[str, object]:
    whole, _, fraction = device.datasheet_fp32_tflops.partition(".")
    published = int(whole) * 10**12 + int(fraction or "0") * 10 ** (12 - len(fraction))
    half_unit = 5 * 10 ** (11 - len(fraction))
    derived = (
        device.sm_count
        * device.architecture.fp32_lanes_per_sm
        * 2
        * device.boost_clock_mhz
        * 10**6
    )
    difference = abs(derived - published)
    return {
        "published_tflops": device.datasheet_fp32_tflops,
        "published_flops_per_second": published,
        "rounding_half_unit_flops_per_second": half_unit,
        "derived_flops_per_second": derived,
        "absolute_difference_flops_per_second": difference,
        "within_published_rounding": difference <= half_unit,
    }


def _peak(device: SimtDevice) -> int:
    return _datasheet(device)["derived_flops_per_second"]


def _bandwidth(device: SimtDevice) -> int:
    return device.dram_bandwidth_gb_per_second * 10**9


def _launch(
    architecture: SimtArchitecture, grid: list[int], block: list[int]
) -> dict[str, object]:
    threads = block[0] * block[1] * block[2]
    blocks = grid[0] * grid[1] * grid[2]
    warps = _ceil(threads, architecture.warp_size)
    idle = warps * architecture.warp_size - threads
    return {
        "grid": grid,
        "block": block,
        "threads_per_block": threads,
        "warps_per_block": warps,
        "idle_lanes_per_block": idle,
        "total_blocks": blocks,
        "total_threads": threads * blocks,
        "total_warps": warps * blocks,
        "total_idle_lanes": idle * blocks,
        "lane_utilization": threads / (warps * architecture.warp_size),
    }


def _warp_classes_1d(
    blocks: int, block: int, elements: int, warp_size: int
) -> tuple[int, int, int]:
    warps = _ceil(block, warp_size)
    full = min(blocks, elements // block)
    valid = elements - full * block if full < blocks else 0
    active = full * warps + valid // warp_size
    partial = 1 if valid % warp_size else 0
    return active, partial, blocks * warps - active - partial


def _warp_classes_2d(
    grid: list[int], block: list[int], width: int, height: int, warp_size: int
) -> tuple[int, int, int]:
    columns = Counter(
        max(0, min(block[0], width - index * block[0])) for index in range(grid[0])
    )
    rows = Counter(
        max(0, min(block[1], height - index * block[1])) for index in range(grid[1])
    )
    threads = block[0] * block[1]
    totals = [0, 0, 0]
    for (valid_x, count_x), (valid_y, count_y) in (
        (x, y) for x in sorted(columns.items()) for y in sorted(rows.items())
    ):
        for first in range(0, threads, warp_size):
            lanes = [
                divmod(thread, block[0])
                for thread in range(first, min(first + warp_size, threads))
            ]
            passing = sum(1 for y, x in lanes if x < valid_x and y < valid_y)
            slot = 0 if passing == len(lanes) else 1 if passing else 2
            totals[slot] += count_x * count_y
    return totals[0], totals[1], totals[2]


def _coverage(
    layout: str,
    grid: list[int],
    block: list[int],
    width: int,
    height: int,
    warp_size: int,
) -> dict[str, object]:
    columns = grid[0] * block[0]
    rows = grid[1] * block[1]
    covered = min(columns, width) * min(rows, height)
    launched = columns * rows
    masked = launched - covered
    out_of_bounds = max(0, rows - height) * columns
    for row in range(max(0, height - columns // width - 1), min(rows, height)):
        out_of_bounds += max(0, columns - (height - row) * width)
    if layout == "1d":
        warps = _warp_classes_1d(grid[0], block[0], width, warp_size)
    else:
        warps = _warp_classes_2d(grid, block, width, height, warp_size)
    return {
        "layout": layout,
        "width": width,
        "height": height,
        "elements": width * height,
        "launched_threads": launched,
        "covered_elements": covered,
        "uncovered_elements": width * height - covered,
        "masked_threads": masked,
        "unguarded_out_of_bounds_writes": out_of_bounds,
        "unguarded_aliased_writes": masked - out_of_bounds,
        "fully_active_warps": warps[0],
        "partially_masked_warps": warps[1],
        "fully_masked_warps": warps[2],
        "under_covered": covered < width * height,
        "needs_bounds_guard": masked > 0,
    }


def _occupancy(
    architecture: SimtArchitecture,
    threads: int,
    registers: int,
    static: int = 0,
    dynamic: int = 0,
) -> tuple[dict[str, object] | None, list[str]]:
    warp = architecture.warp_size
    warps = _ceil(threads, warp)
    unit = architecture.register_allocation_unit
    partitions = architecture.register_sub_partitions_per_sm
    per_warp = _ceil(registers * warp, unit) * unit
    per_block = per_warp * warps
    hardware = per_warp * _ceil(warps, partitions) * partitions
    per_partition = architecture.registers_per_sm // partitions // per_warp
    blockers = [
        name
        for name, failed in (
            ("registers_per_thread", registers > architecture.max_registers_per_thread),
            ("registers_per_block", per_block > architecture.registers_per_block),
            (
                "registers_per_block_partition_check",
                hardware > architecture.registers_per_block,
            ),
        )
        if failed
    ]
    register_blocks = 0 if blockers else per_partition * partitions // warps
    reserved = architecture.reserved_shared_memory_per_block_bytes
    requested = static + dynamic + reserved
    granule = architecture.shared_memory_allocation_unit_bytes
    allocated = _ceil(requested, granule) * granule
    limit = architecture.shared_memory_per_block_bytes + (
        reserved if architecture.major >= 8 else 0
    )
    carveout = architecture.shared_memory_per_sm_bytes
    shared_blocks: int | None
    if allocated > limit:
        blockers.append("shared_memory_per_block")
        shared_blocks = 0
    else:
        shared_blocks = carveout // allocated if allocated else None
    if blockers:
        return None, blockers
    limits = {
        "warps": architecture.max_warps_per_sm // warps,
        "registers": register_blocks,
        "shared_memory": shared_blocks,
        "blocks": architecture.max_blocks_per_sm,
    }
    active = min(value for value in limits.values() if value is not None)
    return {
        "compute_capability": architecture.compute_capability,
        "kernel": {
            "threads_per_block": threads,
            "registers_per_thread": registers,
            "static_shared_bytes": static,
            "dynamic_shared_bytes": dynamic,
            "shared_memory_opt_in": False,
        },
        "warps_per_block": warps,
        "limits_blocks_per_sm": limits,
        "registers": {
            "registers_per_thread": registers,
            "warps_per_block": warps,
            "registers_per_warp": per_warp,
            "registers_per_block": per_block,
            "hardware_check_registers_per_block": hardware,
            "warps_per_sub_partition": per_partition,
            "blocks": register_blocks,
            "blockers": [],
        },
        "shared_memory": {
            "requested_bytes_per_block": requested,
            "allocated_bytes_per_block": allocated,
            "per_block_limit_bytes": limit,
            "configured_carveout_bytes": carveout,
            "effective_carveout_bytes": carveout,
            "blocks": shared_blocks,
            "blockers": [],
        },
        "active_blocks_per_sm": active,
        "active_warps_per_sm": active * warps,
        "max_warps_per_sm": architecture.max_warps_per_sm,
        "occupancy": active * warps / architecture.max_warps_per_sm,
        "limiting_factors": [name for name in _LIMITS if limits[name] == active],
    }, []


def _resident(
    architecture: SimtArchitecture, threads: int, registers: int
) -> dict[str, object]:
    result, blockers = _occupancy(architecture, threads, registers)
    if result is None:
        raise SimtVerificationError(f"declared kernel cannot be resident: {blockers}")
    return result


def _waves(device: SimtDevice, blocks: int, active: int) -> dict[str, object]:
    slots = device.sm_count * active
    full, tail = divmod(blocks, slots)
    waves = full + (1 if tail else 0)
    return {
        "device": device.key,
        "total_blocks": blocks,
        "sm_count": device.sm_count,
        "active_blocks_per_sm": active,
        "slots_per_wave": slots,
        "full_waves": full,
        "tail_blocks": tail,
        "waves": waves,
        "last_wave_fill": tail / slots if tail else 1.0,
        "sm_slot_utilization": blocks / (waves * slots),
    }


def _workload(kind: str, flops: int, read: int, written: int) -> dict[str, object]:
    return {
        "kind": kind,
        "flops": flops,
        "bytes_read": read,
        "bytes_written": written,
        "total_bytes": read + written,
        "arithmetic_intensity": flops / (read + written),
    }


def _roofline(workload: dict[str, object], device: SimtDevice) -> dict[str, object]:
    peak = float(_peak(device))
    bandwidth = float(_bandwidth(device))
    compute = workload["flops"] / peak
    memory = workload["total_bytes"] / bandwidth
    return {
        "bottleneck": "compute"
        if compute > memory
        else "memory"
        if memory > compute
        else "ridge",
        "compute_seconds": compute,
        "memory_seconds": memory,
        "lower_bound_seconds": max(compute, memory),
        "attainable_flops_per_second": min(
            peak, bandwidth * workload["arithmetic_intensity"]
        ),
    }


def _expected_sections(plan: SimtPlan) -> dict[str, object]:
    devices = {
        device.key: {
            "name": device.name,
            "compute_capability": device.compute_capability,
            "fp32_datasheet_check": _datasheet(device),
            "throughput": {
                "device": device.key,
                "peak_fp32_flops_per_second": _peak(device),
                "dram_bandwidth_bytes_per_second": _bandwidth(device),
                "ridge_point_flops_per_byte": _peak(device) / _bandwidth(device),
            },
        }
        for device in plan.devices
    }
    vector = plan.vector_add
    width = DTYPE_BYTES[vector.dtype]
    vector_workload = _workload(
        "vector_add",
        vector.elements,
        2 * vector.elements * width,
        vector.elements * width,
    )
    vector_devices = {}
    for key in vector.devices:
        device = plan.device(key)
        grid = [_ceil(vector.elements, vector.threads_per_block), 1, 1]
        block = [vector.threads_per_block, 1, 1]
        occupancy = _resident(
            device.architecture, vector.threads_per_block, vector.registers_per_thread
        )
        vector_devices[key] = {
            "launch": _launch(device.architecture, grid, block),
            "coverage": _coverage(
                "1d", grid, block, vector.elements, 1, device.architecture.warp_size
            ),
            "occupancy": occupancy,
            "waves": _waves(device, grid[0], occupancy["active_blocks_per_sm"]),
            "roofline": _roofline(vector_workload, device),
        }
    image = plan.rgb_to_grayscale
    pixels = image.width * image.height
    image_workload = _workload(
        "rgb_to_grayscale",
        pixels * image.flops_per_pixel,
        pixels * image.input_bytes_per_pixel,
        pixels * image.output_bytes_per_pixel,
    )
    image_devices = {}
    for key in image.devices:
        device = plan.device(key)
        warp = device.architecture.warp_size
        block = [image.block_x, image.block_y, 1]
        grid = [
            _ceil(image.width, image.block_x),
            _ceil(image.height, image.block_y),
            1,
        ]
        floor = [
            max(1, image.width // image.block_x),
            max(1, image.height // image.block_y),
            1,
        ]
        occupancy = _resident(
            device.architecture,
            image.block_x * image.block_y,
            image.registers_per_thread,
        )
        image_devices[key] = {
            "launch": _launch(device.architecture, grid, block),
            "coverage": _coverage(
                "2d_row_major", grid, block, image.width, image.height, warp
            ),
            "floor_division_launch": _launch(device.architecture, floor, block),
            "floor_division_coverage": _coverage(
                "2d_row_major", floor, block, image.width, image.height, warp
            ),
            "occupancy": occupancy,
            "waves": _waves(
                device, grid[0] * grid[1], occupancy["active_blocks_per_sm"]
            ),
            "roofline": _roofline(image_workload, device),
        }
    sweep = plan.occupancy_sweep
    cells_by_device = {}
    for key in sweep.devices:
        architecture = plan.device(key).architecture
        cells = []
        for threads in sweep.threads_per_block:
            for registers in sweep.registers_per_thread:
                result, blockers = _occupancy(
                    architecture,
                    threads,
                    registers,
                    sweep.static_shared_bytes,
                    sweep.dynamic_shared_bytes,
                )
                cell: dict[str, object] = {
                    "threads_per_block": threads,
                    "registers_per_thread": registers,
                }
                if result is None:
                    cell.update(resident=False, blockers=blockers)
                else:
                    cell.update(
                        resident=True,
                        limits_blocks_per_sm=result["limits_blocks_per_sm"],
                        active_blocks_per_sm=result["active_blocks_per_sm"],
                        active_warps_per_sm=result["active_warps_per_sm"],
                        occupancy=result["occupancy"],
                        limiting_factors=result["limiting_factors"],
                    )
                cells.append(cell)
        cells_by_device[key] = cells
    memory_rows = []
    for key in plan.memory_latency.devices:
        device = plan.device(key)
        per_cycle_denominator = device.sm_count * device.boost_clock_mhz * 10**6
        resident = device.architecture.max_warps_per_sm
        for latency in plan.memory_latency.latency_cycles:
            in_flight = Fraction(_bandwidth(device) * latency, per_cycle_denominator)
            for spec in plan.memory_latency.access_patterns:
                per_warp = (
                    spec.bytes_per_lane * spec.active_lanes * spec.independent_loads
                )
                needed = _ceil(
                    _bandwidth(device) * latency, per_cycle_denominator * per_warp
                )
                fraction = min(Fraction(1), resident * per_warp / in_flight)
                memory_rows.append(
                    {
                        "pattern": spec.name,
                        "device": key,
                        "latency_cycles": latency,
                        "latency_seconds": float(
                            Fraction(latency, device.boost_clock_mhz * 10**6)
                        ),
                        "access": {
                            "bytes_per_lane": spec.bytes_per_lane,
                            "active_lanes": spec.active_lanes,
                            "independent_loads": spec.independent_loads,
                        },
                        "bytes_per_warp": per_warp,
                        "bytes_per_cycle_per_sm": float(
                            Fraction(_bandwidth(device), per_cycle_denominator)
                        ),
                        "bytes_in_flight_per_sm": float(in_flight),
                        "warps_needed": needed,
                        "max_warps_per_sm": resident,
                        "full_bandwidth_within_warp_limit": needed <= resident,
                        "resident_warps": resident,
                        "sustainable_bandwidth_fraction": float(fraction),
                        "sustainable_bandwidth_bytes_per_second": float(
                            fraction * _bandwidth(device)
                        ),
                    }
                )
    arithmetic_rows = []
    arithmetic = plan.arithmetic_latency
    for capability in arithmetic.architectures:
        architecture = plan.architecture(capability)
        resident = architecture.max_warps_per_sm
        for pipeline in arithmetic.pipelines:
            rate = (
                Fraction(architecture.fp32_lanes_per_sm, architecture.warp_size)
                if pipeline == "fp32"
                else Fraction(architecture.warp_schedulers_per_sm)
            )
            in_flight = rate * arithmetic.latency_cycles
            for parallelism in arithmetic.independent_instructions:
                needed = _ceil(in_flight.numerator, in_flight.denominator * parallelism)
                arithmetic_rows.append(
                    {
                        "compute_capability": capability,
                        "pipeline": pipeline,
                        "warp_instructions_per_cycle": float(rate),
                        "latency_cycles": arithmetic.latency_cycles,
                        "independent_instructions": parallelism,
                        "instructions_in_flight": float(in_flight),
                        "warps_needed": needed,
                        "max_warps_per_sm": resident,
                        "full_throughput_within_warp_limit": needed <= resident,
                        "resident_warps": resident,
                        "sustainable_throughput_fraction": float(
                            min(Fraction(1), resident * parallelism / in_flight)
                        ),
                    }
                )
    grid = plan.wave_quantization
    waves = {}
    for key in grid.devices:
        device = plan.device(key)
        occupancy = _resident(
            device.architecture, grid.threads_per_block, grid.registers_per_thread
        )
        waves[key] = {
            "occupancy": occupancy,
            "waves": _waves(
                device, grid.total_blocks, occupancy["active_blocks_per_sm"]
            ),
        }
    cells = [cell for rows in cells_by_device.values() for cell in rows]
    return {
        "devices": devices,
        "vector_add": {"workload": vector_workload, "devices": vector_devices},
        "rgb_to_grayscale": {"workload": image_workload, "devices": image_devices},
        "occupancy_sweep": cells_by_device,
        "latency_hiding": {"memory": memory_rows, "arithmetic": arithmetic_rows},
        "wave_quantization": waves,
        "summary": {
            "devices": len(plan.devices),
            "occupancy_sweep_cells": len(cells),
            "non_resident_sweep_cells": sum(not cell["resident"] for cell in cells),
            "memory_latency_rows": len(memory_rows),
            "memory_rows_beyond_warp_limit": sum(
                not row["full_bandwidth_within_warp_limit"] for row in memory_rows
            ),
            "arithmetic_latency_rows": len(arithmetic_rows),
        },
    }


def verify_simt_report(report: object) -> SimtPlan:
    """Return the declared plan only if every report field reproduces exactly."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise SimtVerificationError("SIMT report fields do not match the schema")
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise SimtVerificationError("unsupported SIMT report schema_version")
    if report["kind"] != SIMT_REPORT_KIND or report["measured"] is not False:
        raise SimtVerificationError("SIMT report must remain an unmeasured analysis")
    if report["evidence_boundary"] != dict(SIMT_EVIDENCE_BOUNDARY):
        raise SimtVerificationError("SIMT evidence boundary was weakened or changed")
    if report["exclusions"] != list(SIMT_EXCLUSIONS):
        raise SimtVerificationError("SIMT model exclusions were changed")
    integrity = report["integrity"]
    if not isinstance(integrity, Mapping) or dict(integrity) != {
        "algorithm": "sha256",
        "digest": simt_report_digest(report),
    }:
        raise SimtVerificationError("SIMT report integrity digest does not match")
    try:
        plan = SimtPlan.from_mapping(report["declared_inputs"])
    except (SimtError, TypeError) as error:
        raise SimtVerificationError(f"declared inputs are invalid: {error}") from error
    _compare(plan.to_mapping(), report["declared_inputs"], "declared_inputs")
    expected = _expected_sections(plan)
    for name, section in expected.items():
        _compare(section, report[name], name)
    rebuilt = seal_simt_report(build_simt_report(plan))
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(report)):
        raise SimtVerificationError("SIMT report differs from a fresh build")
    return plan
