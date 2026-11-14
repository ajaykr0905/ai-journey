"""Canonical sealed reports for the memory-access lab.

Two report kinds share one sealing scheme and never mix evidence:

- the analytical report recomputes every pattern, transpose, tile-traffic and
  roofline number from the declared plan; nothing in it was measured;
- the measurement report records real wall-clock CPU matmul samples taken on
  the host that built it, with the checks that make them interpretable. When
  an injected clock produced the samples, the same report is labeled
  synthetic instead: timer ``injected``, ``measured: false`` and a boundary
  that denies any host measurement.

Each report carries an explicit evidence boundary. The SHA-256 seal covers
canonical JSON (sorted keys, compact separators, ASCII escapes) of every other
field; it detects drift and tampering but is not a signature.
"""

from __future__ import annotations

from hashlib import sha256

from .memory_access_banks import MODELED_BANK_ACCESS_BYTES, analyze_banks
from .memory_access_baseline import MatmulMeasurement, cpu_environment
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_coalescing import analyze_coalescing
from .memory_access_pattern import MemoryAccessError
from .memory_access_plan import MemoryAccessPlan, PatternSpec, canonical_json_bytes
from .memory_access_roofline import roofline_estimate_mapping, sweep_tile_rooflines
from .memory_access_tile_banks import analyze_inner_loop_banks
from .memory_access_tiling import (
    MatmulShape,
    NaiveMatmulTraffic,
    matmul_element_bytes,
)

REPORT_SCHEMA_VERSION = 1
ANALYSIS_KIND = "memory_access_analysis"
MEASUREMENT_KIND = "memory_access_cpu_measurement"
MEASUREMENT_SCOPE = "cpu_wall_clock_this_host"
SYNTHETIC_SCOPE = "synthetic_injected_clock"
DIGEST_FIELD = "report_sha256"

ANALYSIS_BOUNDARY = {
    "analytical_only": True,
    "gpu_executed": False,
    "cuda_kernel_executed": False,
    "timing_measured": False,
    "hardware_ceilings_declared": True,
    "statement": (
        "Analytical memory-access model computed from declared inputs. No GPU, "
        "CUDA kernel, cache, or device timing was executed or measured; roofline "
        "ceilings are declared datasheet peaks and lower-bound times are "
        "roofline arithmetic."
    ),
}
MEASUREMENT_BOUNDARY = {
    "cpu_measurement_this_host": True,
    "gpu_executed": False,
    "cuda_kernel_executed": False,
    "portable_performance_claim": False,
    "statement": (
        "Wall-clock float32 CPU matmul measurements taken on the host that built "
        "this report. They do not measure or predict GPU, CUDA, tensor-core, "
        "distributed, or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "cpu_measurement_this_host": False,
    "synthetic_timing": True,
    "gpu_executed": False,
    "cuda_kernel_executed": False,
    "portable_performance_claim": False,
    "statement": (
        "Durations came from a clock injected by the caller instead of "
        "time.perf_counter_ns, so they are synthetic values and not a "
        "measurement of this or any host. The float32 CPU matmuls and their "
        "float64 checks did run; no GPU or CUDA kernel was executed."
    ),
}
MODEL_SOURCES = (
    {
        "id": "cuda-cpp-best-practices-guide-v13.5",
        "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
        "sections": [
            "10.2.1 coalesced access to global memory",
            "10.2.1.2 sequential but misaligned access",
            "10.2.1.4 strided accesses",
            "10.2.3.1 shared memory and memory banks",
            "10.2.3.2 shared memory in matrix multiplication",
            "10.2.3.3 padded transposed tile",
        ],
    },
    {
        "id": "cuda-cpp-programming-guide-v12.6",
        "url": (
            "https://docs.nvidia.com/cuda/archive/12.6.0/"
            "cuda-c-programming-guide/index.html"
        ),
        "sections": [
            "2.2 thread hierarchy",
            "4.1 SIMT architecture",
            "5.3.2 device memory accesses",
            "16.2 technical specifications",
            "16.4.3 compute capability 5.x shared memory",
            "16.5.3 and 16.6.4 later shared memory matches 5.x",
        ],
    },
)
CORRECTNESS_SOURCE = (
    "N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., "
    "SIAM, 2002, section 3.1: |fl(x.y) - x.y| <= gamma_K |x|.|y|"
)
EXCLUSIONS = (
    "L1 and L2 caching, hits, evictions, and reuse between warp requests",
    "hardware prefetching",
    "TLB and virtual-address translation",
    "64-bit shared-memory bank modes and 8- or 16-byte shared-memory accesses",
    "asynchronous global-to-shared copies such as cp.async and TMA",
    "tensor cores and register blocking",
    "occupancy, launch geometry, and latency hiding",
)


def report_digest(report: dict[str, object]) -> str:
    """Hash canonical JSON of every field except the digest itself."""

    if type(report) is not dict:
        raise TypeError("report must be a dictionary")
    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    return sha256(canonical_json_bytes(unsigned)).hexdigest()


def seal_report(report: dict[str, object]) -> dict[str, object]:
    """Return a copy carrying a SHA-256 digest over all other fields."""

    if type(report) is not dict:
        raise TypeError("report must be a dictionary")
    if DIGEST_FIELD in report:
        raise MemoryAccessError("report is already sealed")
    return {**report, DIGEST_FIELD: report_digest(report)}


def _pattern_entry(pattern: PatternSpec) -> dict[str, object]:
    access = pattern.build()
    shared = (
        analyze_banks(access).to_mapping()
        if access.access_bytes in MODELED_BANK_ACCESS_BYTES
        else None
    )
    return {
        "name": pattern.name,
        "kind": pattern.kind,
        "access": access.to_mapping(),
        "global_memory": analyze_coalescing(access).to_mapping(),
        "shared_memory": shared,
    }


def _sweep_entry(plan: MemoryAccessPlan, shape: MatmulShape) -> dict[str, object]:
    sweep = sweep_tile_rooflines(
        shape,
        tiles=plan.matmul.tiles,
        hardware=plan.roofline.hardware,
        dtype=plan.matmul.dtype,
    )
    return {
        "shape": shape.to_mapping(),
        "compulsory": roofline_estimate_mapping(sweep.compulsory),
        "naive": {
            "traffic": NaiveMatmulTraffic(shape, plan.matmul.dtype).to_mapping(),
            "roofline": roofline_estimate_mapping(sweep.naive),
        },
        "tiled": [
            {
                "traffic": item.traffic.to_mapping(),
                "roofline": roofline_estimate_mapping(item.estimate),
            }
            for item in sweep.tiles
        ],
        "bottleneck_counts": sweep.bottleneck_counts,
        "first_compute_bound_tile": sweep.first_compute_bound_tile,
    }


def build_analysis_report(plan: MemoryAccessPlan) -> dict[str, object]:
    """Recompute every analytical number from one declared plan."""

    if not isinstance(plan, MemoryAccessPlan):
        raise TypeError("plan must be a MemoryAccessPlan")
    patterns = [_pattern_entry(pattern) for pattern in plan.patterns]
    transposes = [
        {"name": item.name, **item.analyze().to_mapping()} for item in plan.transposes
    ]
    element_bytes = matmul_element_bytes(plan.matmul.dtype)
    inner_loop = (
        [
            analyze_inner_loop_banks(tile, element_bytes=element_bytes).to_mapping()
            for tile in plan.matmul.tiles
        ]
        if element_bytes in MODELED_BANK_ACCESS_BYTES
        else None
    )
    sweeps = [_sweep_entry(plan, shape) for shape in plan.matmul.shapes]
    counts = {"compute": 0, "memory": 0, "ridge": 0}
    for sweep in sweeps:
        for name, count in sweep["bottleneck_counts"].items():
            counts[name] += count
    hardware = plan.roofline.hardware
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "measured": False,
        "evidence_boundary": dict(ANALYSIS_BOUNDARY),
        "plan": plan.to_mapping(),
        "plan_sha256": plan.fingerprint(),
        "model_sources": [dict(source) for source in MODEL_SOURCES],
        "exclusions": list(EXCLUSIONS),
        "patterns": patterns,
        "transposes": transposes,
        "matmul": {
            "dtype": plan.matmul.dtype,
            "roofline": {
                **plan.roofline.to_mapping(),
                "ridge_point_flops_per_byte": hardware.ridge_point,
            },
            "inner_loop_banks": inner_loop,
            "sweeps": sweeps,
        },
        "summary": {
            "patterns": len(patterns),
            "shared_memory_patterns": sum(
                1 for item in patterns if item["shared_memory"] is not None
            ),
            "transposes": len(transposes),
            "matmul_shapes": len(sweeps),
            "tiles": len(plan.matmul.tiles),
            "tiled_estimates": len(sweeps) * len(plan.matmul.tiles),
            "tiled_bottleneck_counts": counts,
        },
    }


def build_measurement_report(
    plan: MemoryAccessPlan,
    measurements: tuple[MatmulMeasurement, ...],
    environment: dict[str, object],
) -> dict[str, object]:
    """Record CPU samples for exactly the plan's benchmark cases.

    Only samples taken with the host timer are labeled a measurement; samples
    from an injected clock get the synthetic timer, scope and boundary.
    """

    if not isinstance(plan, MemoryAccessPlan):
        raise TypeError("plan must be a MemoryAccessPlan")
    if type(measurements) is not tuple or any(
        not isinstance(item, MatmulMeasurement) for item in measurements
    ):
        raise TypeError("measurements must be a tuple of MatmulMeasurement")
    benchmark = plan.benchmark
    if tuple(item.case for item in measurements) != benchmark.cases():
        raise MemoryAccessError("measurements must cover the plan's cases in order")
    if any(
        (item.warmup, item.repeats, item.seed)
        != (benchmark.warmup, benchmark.repeats, benchmark.seed)
        for item in measurements
    ):
        raise MemoryAccessError("measurements must use the plan's benchmark budget")
    if type(environment) is not dict or set(environment) != set(cpu_environment()):
        raise MemoryAccessError("environment fields do not match the CPU record")
    timers = {item.timer for item in measurements}
    if len(timers) != 1:
        raise MemoryAccessError("measurements must share one timer")
    timer = timers.pop()
    if timer not in (TIMER, INJECTED_TIMER):
        raise MemoryAccessError("measurement timer is not recognized")
    measured = timer == TIMER
    results = [item.to_mapping() for item in measurements]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": MEASUREMENT_KIND,
        "measured": measured,
        "measurement_scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
        "evidence_boundary": dict(
            MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY
        ),
        "plan_name": plan.name,
        "plan_sha256": plan.fingerprint(),
        "benchmark": benchmark.to_mapping(),
        "timer": timer,
        "percentile_method": PERCENTILE_METHOD,
        "correctness_source": CORRECTNESS_SOURCE,
        "environment": dict(environment),
        "results": results,
        "summary": {
            "cases": len(results),
            "checked_results": sum(
                item.correctness.checked_results for item in measurements
            ),
            "max_bound_ratio": max(
                item.correctness.max_bound_ratio for item in measurements
            ),
        },
    }
