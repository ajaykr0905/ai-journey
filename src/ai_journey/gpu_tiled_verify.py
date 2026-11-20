"""Independent verification of sealed tiled CUDA contract reports.

Verification trusts no number in a report. It checks the field set, the
fixed evidence boundary and the SHA-256 seal, parses the embedded plan, and
can bind the report to a plan file. It then recomputes each section from
the plan, with other code than the run's where the run used lab code:

* the operands, the float64 reference, the error bound and the fma replay
  with the Day 74 Metal verifier's code. Its replay rounds to odd in float64,
  and the run's replay repairs midpoints, so the two methods check each
  other;
* the grid, the phases, the global loads, the zero-filled slots and the
  barriers by closed forms, with no call to the simulator or to
  ``TiledMatmulTraffic``;
* the occupancy by the register and shared memory rules, with the compute
  capability 8.0 values that the Day 74 model verifier pins.

The source texts must be the pinned texts, and both checkers must pass each
text for each case. The summary follows from the plan alone, so it is
checked before the cases. Last, the block simulator runs each case at each
tile again, and each run must equal the recomputed record. The first field
that differs is named.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .gpu_tiled_blocksim import run_tiled_blocks
from .gpu_tiled_config import GpuTiledPlan
from .gpu_tiled_cuda import (
    CUBLAS_CHECK_NAMES,
    TF32_ASSUMPTION,
    TILED_CHECK_NAMES,
    GpuTiledError,
    check_cublas_source,
    check_tiled_source,
    cublas_source,
    tiled_cuda_source,
)
from .gpu_tiled_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    REPORT_KIND,
    SCHEMA_VERSION,
    report_digest,
    rule_mapping,
)
from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_report import limits_mapping
from .kernel_launch_verify import (
    KernelLaunchVerificationError,
    _ceil,
    _compare,
    _geometry,
)
from .matmul_metal_verify import (
    _bound,
    _hash,
    _operands,
    _reference,
    _relative_error,
    _replay,
)
from .matmul_model_occupancy import architecture_mapping
from .matmul_model_verify import (
    _MAX_BLOCKS_PER_SM,
    _MAX_WARPS_PER_SM,
    _REGISTER_UNIT,
    _REGISTERS_PER_SM,
    _SHARED_PER_BLOCK,
    _SHARED_PER_SM,
    _SHARED_RESERVED,
    _SHARED_UNIT,
    _SUB_PARTITIONS,
    _ratio,
)
from .memory_access_tiling import MatmulShape

MAX_REPORT_BYTES = 1024 * 1024
# Pinned here independently of the modules that the report uses.
_WARP = 32
_FLOAT = 4
_STEPS = ["load", "barrier", "fma", "barrier"]
_CUBLAS_CALL = {
    "transa": "CUBLAS_OP_N",
    "transb": "CUBLAS_OP_N",
    "m": "n",
    "n": "m",
    "k": "k",
    "a": "b",
    "lda": "n",
    "b": "a",
    "ldb": "k",
    "c": "c",
    "ldc": "n",
}
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "plan",
    "limits",
    "rule",
    "sources",
    "cases",
    "occupancy",
    "summary",
    "evidence_boundary",
    DIGEST_FIELD,
}


class GpuTiledVerificationError(GpuTiledError):
    """Raised when a report does not reproduce from its declared plan."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuTiledVerificationError(message)


def _expected_sources(plan: GpuTiledPlan) -> dict[str, object]:
    cublas = cublas_source()
    return {
        "language": "CUDA C++",
        "compiled": False,
        "executed": False,
        "tiled": [
            {
                "tile": tile,
                "kernel": f"tiled_matmul_t{tile}_kernel",
                "launcher": f"launch_tiled_matmul_t{tile}",
                "shared_memory_bytes": 2 * tile * tile * _FLOAT,
                "sha256": sha256(tiled_cuda_source(tile=tile).encode()).hexdigest(),
                "text": tiled_cuda_source(tile=tile),
            }
            for tile in plan.tiles
        ],
        "cublas": {
            "function": "sgemm_row_major",
            "header": "#include <cublas_v2.h>",
            "called": False,
            "tf32_assumption": TF32_ASSUMPTION,
            "sha256": sha256(cublas.encode()).hexdigest(),
            "text": cublas,
        },
    }


def _expected_estimate(tile: int, registers: int) -> dict[str, object]:
    threads = tile * tile
    warps = _ceil(threads, _WARP)
    per_warp = _ceil(registers * _WARP, _REGISTER_UNIT) * _REGISTER_UNIT
    per_sub_partition = _REGISTERS_PER_SM // _SUB_PARTITIONS // per_warp
    register_blocks = per_sub_partition * _SUB_PARTITIONS // warps
    static = 2 * tile * tile * _FLOAT
    requested = static + _SHARED_RESERVED
    allocated = _ceil(requested, _SHARED_UNIT) * _SHARED_UNIT
    limits = {
        "warps": _MAX_WARPS_PER_SM // warps,
        "registers": register_blocks,
        "shared_memory": _SHARED_PER_SM // allocated,
        "blocks": _MAX_BLOCKS_PER_SM,
    }
    active = min(limits.values())
    return {
        "tile": tile,
        "compute_capability": "8.0",
        "kernel": {
            "threads_per_block": threads,
            "registers_per_thread": registers,
            "static_shared_bytes": static,
            "dynamic_shared_bytes": 0,
            "shared_memory_opt_in": False,
        },
        "warps_per_block": warps,
        "limits_blocks_per_sm": limits,
        "registers": {
            "registers_per_thread": registers,
            "warps_per_block": warps,
            "registers_per_warp": per_warp,
            "registers_per_block": per_warp * warps,
            "hardware_check_registers_per_block": per_warp
            * _ceil(warps, _SUB_PARTITIONS)
            * _SUB_PARTITIONS,
            "warps_per_sub_partition": per_sub_partition,
            "blocks": register_blocks,
            "blockers": [],
        },
        "shared_memory": {
            "requested_bytes_per_block": requested,
            "allocated_bytes_per_block": allocated,
            "per_block_limit_bytes": _SHARED_PER_BLOCK + _SHARED_RESERVED,
            "configured_carveout_bytes": _SHARED_PER_SM,
            "effective_carveout_bytes": _SHARED_PER_SM,
            "blocks": _SHARED_PER_SM // allocated,
            "blockers": [],
        },
        "active_blocks_per_sm": active,
        "active_warps_per_sm": active * warps,
        "max_warps_per_sm": _MAX_WARPS_PER_SM,
        "occupancy": _ratio(Fraction(active * warps, _MAX_WARPS_PER_SM)),
        "limiting_factors": [name for name, value in limits.items() if value == active],
    }


def _expected_summary(plan: GpuTiledPlan) -> dict[str, object]:
    cases, tiles = len(plan.shapes), len(plan.tiles)
    runs = cases * tiles
    return {
        "cases": cases,
        "tiles": tiles,
        "block_runs": runs,
        "block_runs_matching_replay": runs,
        "block_runs_matching_traffic": runs,
        "race_free_block_runs": runs,
        "stored_once_block_runs": runs,
        "cuda_checks_passed": len(TILED_CHECK_NAMES) * runs,
        "cuda_checks": len(TILED_CHECK_NAMES) * runs,
        "cublas_checks_passed": len(CUBLAS_CHECK_NAMES) * cases,
        "cublas_checks": len(CUBLAS_CHECK_NAMES) * cases,
        "occupancy_estimates": tiles * len(plan.registers_per_thread),
    }


def _expected_run(tile: int, shape: MatmulShape, replay: str) -> dict[str, object]:
    """Return a tile run by closed forms: one block per T x T tile of C."""

    m, n, k = shape.m, shape.n, shape.k
    grid = [_ceil(n, tile), _ceil(m, tile), 1]
    phases = _ceil(k, tile)
    slots = 2 * tile * tile * phases * grid[0] * grid[1]
    # Each block column reads all of A once, and each block row all of B.
    a_loads, b_loads = grid[0] * m * k, grid[1] * k * n
    return {
        "tile": tile,
        "shape": {"m": m, "n": n, "k": k},
        "launch": _geometry(grid, [tile, tile, 1]),
        "steps": list(_STEPS),
        "phases": phases,
        "inner_steps": k,
        "global_loads": {
            "a": a_loads,
            "b": b_loads,
            "zero_filled_slots": slots - a_loads - b_loads,
            "tile_slot_writes": slots,
        },
        "stores": m * n,
        "stored_once": True,
        "matches_traffic": True,
        "hazards": {
            "barriers": 2 * phases,
            "read_after_write": 0,
            "write_after_read": 0,
            "race_free": True,
        },
        "output_sha256": replay,
    }


def _verify_case(
    plan: GpuTiledPlan, index: int, shape: MatmulShape, actual: object
) -> None:
    """Recompute one case, compare it, then run the simulator on it again."""

    m, n, k = shape.m, shape.n, shape.k
    left, right = _operands(plan.seed, index, m, n, k)
    replay = _replay(left, right)
    total, scale = _reference(left, right)
    replay_digest = _hash(replay)
    error, bound = _relative_error(replay, total, scale), _bound(k)
    _require(error <= bound, f"the replay of case {index} exceeds its error bound")
    tiles = []
    for tile in plan.tiles:
        cuda_check = {
            "tile": tile,
            "checks": dict.fromkeys(TILED_CHECK_NAMES, True),
            "passed": True,
            "shared_memory_bytes": 2 * tile * tile * _FLOAT,
            "launch": {
                "grid": [_ceil(n, tile), _ceil(m, tile), 1],
                "block": [tile, tile, 1],
            },
        }
        found = check_tiled_source(tiled_cuda_source(tile=tile), tile=tile, shape=shape)
        _compare(cuda_check, found.to_mapping(), f"cases[{index}] tile {tile} text")
        tiles.append(
            {
                "tile": tile,
                "cuda_check": cuda_check,
                "run": _expected_run(tile, shape, replay_digest),
                "matches_replay": True,
            }
        )
    cublas_check = {
        "checks": dict.fromkeys(CUBLAS_CHECK_NAMES, True),
        "passed": True,
        "call": dict(_CUBLAS_CALL),
    }
    found = check_cublas_source(cublas_source(), shape=shape)
    _compare(cublas_check, found.to_mapping(), f"cases[{index}] cuBLAS text")
    expected = {
        "index": index,
        "shape": {"m": m, "n": n, "k": k},
        "inputs_sha256": {"a": _hash(left), "b": _hash(right)},
        "replay_sha256": replay_digest,
        "reference_sha256": _hash(total),
        "error_bound": bound,
        "replay_relative_error": error,
        "cublas_check": cublas_check,
        "tiles": tiles,
    }
    _compare(expected, actual, f"cases[{index}]")
    for entry in tiles:
        rerun = run_tiled_blocks(left, right, tile=entry["tile"]).to_mapping()
        _compare(entry["run"], rerun, f"cases[{index}] tile {entry['tile']} rerun")


def _verify(report: Mapping[str, object], plan: GpuTiledPlan | None) -> GpuTiledPlan:
    if type(report["schema_version"]) is not int or (
        report["schema_version"] != SCHEMA_VERSION
    ):
        raise GpuTiledVerificationError("unsupported report schema_version")
    if report["kind"] != REPORT_KIND:
        raise GpuTiledVerificationError(f"report kind is not {REPORT_KIND}")
    if report[DIGEST_FIELD] != report_digest(report):
        raise GpuTiledVerificationError("report digest does not match its content")
    _compare(dict(EVIDENCE_BOUNDARY), report["evidence_boundary"], "evidence_boundary")
    try:
        embedded = GpuTiledPlan.from_mapping(report["plan"])
    except (ValueError, TypeError) as error:
        raise GpuTiledVerificationError(f"embedded plan is invalid: {error}") from error
    _compare(embedded.to_mapping(), report["plan"], "plan")
    if plan is not None:
        if type(plan) is not GpuTiledPlan:
            raise TypeError("plan must be GpuTiledPlan")
        if plan != embedded:
            raise GpuTiledVerificationError("report was not built from this plan")
    cheap = {
        "limits": limits_mapping(),
        "rule": rule_mapping(),
        "sources": _expected_sources(embedded),
        "occupancy": {
            "architecture": architecture_mapping(),
            "estimates": [
                _expected_estimate(tile, count)
                for tile in embedded.tiles
                for count in embedded.registers_per_thread
            ],
        },
        "summary": _expected_summary(embedded),
    }
    for name, section in cheap.items():
        _compare(section, report[name], name)
    cases = report["cases"]
    _require(
        type(cases) is list and len(cases) == len(embedded.shapes),
        "cases rows do not match",
    )
    for index, (shape, actual) in enumerate(zip(embedded.shapes, cases)):
        _verify_case(embedded, index, shape, actual)
    return embedded


def verify_report(report: object, *, plan: GpuTiledPlan | None = None) -> GpuTiledPlan:
    """Return the report's plan only if every field reproduces exactly."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise GpuTiledVerificationError("report fields do not match the schema")
    try:
        return _verify(report, plan)
    except KernelLaunchVerificationError as error:
        # The Day 73 comparison helper names the field; keep its message.
        raise GpuTiledVerificationError(str(error)) from error


def load_report(path: Path, *, plan: GpuTiledPlan | None = None) -> dict[str, object]:
    """Load canonical report bytes and verify every field.

    A file that cannot be read raises ``KernelLaunchError``. A file that was
    read but is not a canonical, reproducible report raises
    ``GpuTiledVerificationError``.
    """

    payload = read_bounded(path, limit=MAX_REPORT_BYTES, label="report")
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise GpuTiledVerificationError(str(error)) from error
    if type(report) is not dict:
        raise GpuTiledVerificationError("report must be a JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise GpuTiledVerificationError("report bytes are not canonical JSON")
    verify_report(report, plan=plan)
    return report
