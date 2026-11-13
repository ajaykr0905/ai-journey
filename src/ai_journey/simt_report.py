"""Canonical sealed report of the analytical SIMT execution model.

The report embeds every declared input, every derived number, an explicit
evidence boundary, and the model's exclusions, then carries a SHA-256 digest
over its canonical JSON. The digest detects drift; it is not a signature.
"""

from __future__ import annotations

from collections.abc import Mapping
import json

from .roofline import KernelWorkload, RooflineEstimate
from .simt_device import SimtError
from .simt_indexing import coverage_1d, coverage_2d
from .simt_json import canonical_json_bytes, canonical_sha256
from .simt_latency import (
    MemoryAccessPattern,
    arithmetic_latency_hiding,
    memory_latency_hiding,
)
from .simt_launch import (
    Dim3,
    LaunchConfiguration,
    check_launch,
    launch_for_elements,
    launch_for_image,
)
from .simt_occupancy import KernelResources, LaunchNotResidentError, analyze_occupancy
from .simt_plan import SimtPlan
from .simt_throughput import (
    device_throughput,
    rgb_to_grayscale_workload,
    roofline_on_device,
    vector_add_workload,
)
from .simt_waves import wave_quantization

SIMT_REPORT_KIND = "analytical_simt_execution_model"
SIMT_EVIDENCE_BOUNDARY: Mapping[str, object] = {
    "analytical_only": True,
    "declared_inputs_only": True,
    "gpu_executed": False,
    "cuda_kernels_compiled": False,
    "latency_measured": False,
    "learner_completion": False,
    "statement": (
        "Calculator predictions from declared public-source inputs. No GPU, CUDA "
        "kernel, or latency was executed or measured to produce them, so none of "
        "these numbers is achieved occupancy, bandwidth, or run time."
    ),
}
SIMT_EXCLUSIONS = (
    "Partitioned global caching: cuda_occupancy.h applies it only to compute "
    "capability 5.2, 5.3, 6.1 and 6.2, none of which is pinned.",
    "Thread block clusters and distributed shared memory (compute capability 9.0 "
    "and later).",
    "Block-barrier occupancy limits, which cuda_occupancy.h applies from compute "
    "capability 9.0.",
    "Driver carveout selection heuristics: the configured carveout, or the "
    "maximum when none is configured, is assumed.",
    "L1, L2, texture and constant cache effects, memory coalescing efficiency, "
    "and DRAM efficiency below the declared peak.",
    "Instruction mix, divergence cost, warp scheduler policy, and any latency "
    "other than the declared ones.",
    "Clock behavior other than the declared boost clock, kernel launch overhead, "
    "and concurrent kernels.",
)


def _workload_mapping(workload: KernelWorkload) -> dict[str, object]:
    return {
        "kind": workload.kind,
        "flops": workload.flops,
        "bytes_read": workload.bytes_read,
        "bytes_written": workload.bytes_written,
        "total_bytes": workload.total_bytes,
        "arithmetic_intensity": workload.arithmetic_intensity,
    }


def _roofline_mapping(estimate: RooflineEstimate) -> dict[str, object]:
    return {
        "bottleneck": estimate.bottleneck,
        "compute_seconds": estimate.compute_seconds,
        "memory_seconds": estimate.memory_seconds,
        "lower_bound_seconds": estimate.lower_bound_seconds,
        "attainable_flops_per_second": estimate.attainable_flops_per_second,
    }


def _resident(plan: SimtPlan, key: str, kernel: KernelResources) -> dict[str, object]:
    try:
        return analyze_occupancy(plan.device(key).architecture, kernel).to_mapping()
    except LaunchNotResidentError as error:
        raise SimtError(f"declared kernel is not resident on {key}: {error}") from error


def _devices_section(plan: SimtPlan) -> dict[str, object]:
    section = {}
    for device in plan.devices:
        check = device.fp32_datasheet_check()
        section[device.key] = {
            "name": device.name,
            "compute_capability": device.compute_capability,
            "fp32_datasheet_check": {
                "published_tflops": check.published_tflops,
                "published_flops_per_second": check.published_flops_per_second,
                "rounding_half_unit_flops_per_second": (
                    check.rounding_half_unit_flops_per_second
                ),
                "derived_flops_per_second": check.derived_flops_per_second,
                "absolute_difference_flops_per_second": (
                    check.absolute_difference_flops_per_second
                ),
                "within_published_rounding": check.within_published_rounding,
            },
            "throughput": device_throughput(device).to_mapping(),
        }
    return section


def _vector_add_section(plan: SimtPlan) -> dict[str, object]:
    scenario = plan.vector_add
    workload = vector_add_workload(elements=scenario.elements, dtype=scenario.dtype)
    per_device = {}
    for key in scenario.devices:
        device = plan.device(key)
        geometry = launch_for_elements(
            device.architecture,
            elements=scenario.elements,
            threads_per_block=scenario.threads_per_block,
        )
        occupancy = _resident(
            plan,
            key,
            KernelResources(scenario.threads_per_block, scenario.registers_per_thread),
        )
        per_device[key] = {
            "launch": geometry.to_mapping(),
            "coverage": coverage_1d(geometry, elements=scenario.elements).to_mapping(),
            "occupancy": occupancy,
            "waves": wave_quantization(
                device,
                total_blocks=geometry.total_blocks,
                active_blocks_per_sm=occupancy["active_blocks_per_sm"],
            ).to_mapping(),
            "roofline": _roofline_mapping(roofline_on_device(workload, device)),
        }
    return {"workload": _workload_mapping(workload), "devices": per_device}


def _grayscale_section(plan: SimtPlan) -> dict[str, object]:
    scenario = plan.rgb_to_grayscale
    workload = rgb_to_grayscale_workload(
        width=scenario.width,
        height=scenario.height,
        input_bytes_per_pixel=scenario.input_bytes_per_pixel,
        output_bytes_per_pixel=scenario.output_bytes_per_pixel,
        flops_per_pixel=scenario.flops_per_pixel,
    )
    per_device = {}
    for key in scenario.devices:
        device = plan.device(key)
        geometry = launch_for_image(
            device.architecture,
            width=scenario.width,
            height=scenario.height,
            block_x=scenario.block_x,
            block_y=scenario.block_y,
        )
        floor_geometry = check_launch(
            device.architecture,
            LaunchConfiguration(
                Dim3(
                    max(1, scenario.width // scenario.block_x),
                    max(1, scenario.height // scenario.block_y),
                ),
                geometry.launch.block,
            ),
        )
        occupancy = _resident(
            plan,
            key,
            KernelResources(geometry.threads_per_block, scenario.registers_per_thread),
        )
        per_device[key] = {
            "launch": geometry.to_mapping(),
            "coverage": coverage_2d(
                geometry, width=scenario.width, height=scenario.height
            ).to_mapping(),
            "floor_division_launch": floor_geometry.to_mapping(),
            "floor_division_coverage": coverage_2d(
                floor_geometry, width=scenario.width, height=scenario.height
            ).to_mapping(),
            "occupancy": occupancy,
            "waves": wave_quantization(
                device,
                total_blocks=geometry.total_blocks,
                active_blocks_per_sm=occupancy["active_blocks_per_sm"],
            ).to_mapping(),
            "roofline": _roofline_mapping(roofline_on_device(workload, device)),
        }
    return {"workload": _workload_mapping(workload), "devices": per_device}


def _sweep_section(plan: SimtPlan) -> dict[str, object]:
    scenario = plan.occupancy_sweep
    section = {}
    for key in scenario.devices:
        architecture = plan.device(key).architecture
        cells = []
        for threads in scenario.threads_per_block:
            for registers in scenario.registers_per_thread:
                kernel = KernelResources(
                    threads,
                    registers,
                    static_shared_bytes=scenario.static_shared_bytes,
                    dynamic_shared_bytes=scenario.dynamic_shared_bytes,
                )
                cell: dict[str, object] = {
                    "threads_per_block": threads,
                    "registers_per_thread": registers,
                }
                try:
                    result = analyze_occupancy(architecture, kernel).to_mapping()
                except LaunchNotResidentError as error:
                    cell.update(resident=False, blockers=list(error.blockers))
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
        section[key] = cells
    return section


def _latency_section(plan: SimtPlan) -> dict[str, object]:
    memory = []
    for key in plan.memory_latency.devices:
        device = plan.device(key)
        for latency in plan.memory_latency.latency_cycles:
            for spec in plan.memory_latency.access_patterns:
                row = memory_latency_hiding(
                    device,
                    latency_cycles=latency,
                    pattern=MemoryAccessPattern(
                        spec.bytes_per_lane, spec.active_lanes, spec.independent_loads
                    ),
                    resident_warps=device.architecture.max_warps_per_sm,
                ).to_mapping()
                memory.append({"pattern": spec.name, **row})
    arithmetic = []
    scenario = plan.arithmetic_latency
    for capability in scenario.architectures:
        architecture = plan.architecture(capability)
        for pipeline in scenario.pipelines:
            for parallelism in scenario.independent_instructions:
                arithmetic.append(
                    arithmetic_latency_hiding(
                        architecture,
                        pipeline=pipeline,
                        latency_cycles=scenario.latency_cycles,
                        independent_instructions=parallelism,
                        resident_warps=architecture.max_warps_per_sm,
                    ).to_mapping()
                )
    return {"memory": memory, "arithmetic": arithmetic}


def _waves_section(plan: SimtPlan) -> dict[str, object]:
    scenario = plan.wave_quantization
    section = {}
    for key in scenario.devices:
        occupancy = _resident(
            plan,
            key,
            KernelResources(scenario.threads_per_block, scenario.registers_per_thread),
        )
        section[key] = {
            "occupancy": occupancy,
            "waves": wave_quantization(
                plan.device(key),
                total_blocks=scenario.total_blocks,
                active_blocks_per_sm=occupancy["active_blocks_per_sm"],
            ).to_mapping(),
        }
    return section


def build_simt_report(plan: SimtPlan) -> dict[str, object]:
    """Derive every scenario from a validated plan; no GPU is touched."""

    if not isinstance(plan, SimtPlan):
        raise TypeError("plan must be SimtPlan")
    sweep = _sweep_section(plan)
    latency = _latency_section(plan)
    cells = [cell for rows in sweep.values() for cell in rows]
    return {
        "schema_version": 1,
        "kind": SIMT_REPORT_KIND,
        "measured": False,
        "declared_inputs": plan.to_mapping(),
        "devices": _devices_section(plan),
        "vector_add": _vector_add_section(plan),
        "rgb_to_grayscale": _grayscale_section(plan),
        "occupancy_sweep": sweep,
        "latency_hiding": latency,
        "wave_quantization": _waves_section(plan),
        "summary": {
            "devices": len(plan.devices),
            "occupancy_sweep_cells": len(cells),
            "non_resident_sweep_cells": sum(not cell["resident"] for cell in cells),
            "memory_latency_rows": len(latency["memory"]),
            "memory_rows_beyond_warp_limit": sum(
                not row["full_bandwidth_within_warp_limit"] for row in latency["memory"]
            ),
            "arithmetic_latency_rows": len(latency["arithmetic"]),
        },
        "evidence_boundary": dict(SIMT_EVIDENCE_BOUNDARY),
        "exclusions": list(SIMT_EXCLUSIONS),
    }


def simt_report_digest(report: Mapping[str, object]) -> str:
    """Return the SHA-256 of the canonical report without its integrity field."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    return canonical_sha256(
        {key: value for key, value in report.items() if key != "integrity"}
    )


def seal_simt_report(report: Mapping[str, object]) -> dict[str, object]:
    """Return a canonical copy with an integrity digest; the input is unchanged."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    sealed = json.loads(canonical_json_bytes(unsigned))
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": canonical_sha256(unsigned),
    }
    return sealed
