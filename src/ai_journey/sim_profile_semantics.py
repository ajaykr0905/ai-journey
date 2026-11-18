"""A pinned digest of what the simulator records, so optimizations keep it.

A faster simulator must still record every out-of-bounds access, every
write conflict and every read-write conflict, in the same order, for every
launch order. The pinned workloads are guarded and race-free, so they cannot
show a lost record. This module runs four misuse cases instead, each in the
canonical order and in one seeded permutation:

* the unguarded Day 73 vector add on a ragged extent: out-of-bounds reads
  and writes past the end;
* the unguarded Day 73 grayscale kernel on a ragged image: wrapped writes
  that race, and writes past the end;
* a three-point stencil that updates its own input: negative and past-the-end
  indices, and read-write races whose result depends on the order;
* a guarded Day 74 naive matmul with the x_rows mapping: no misuse, and the
  same bits in both orders.

Every record of each log goes into one canonical JSON list. Its SHA-256 was
taken on the simulator before the first optimization round, and each
profiling snapshot refuses a tree whose digest differs.
"""

from __future__ import annotations

from .kernel_launch_buffer import DeviceBuffer, LaunchAccessLog, to_float32
from .kernel_launch_config import launch_1d
from .kernel_launch_io import canonical_sha256
from .kernel_launch_kernels import KernelRun, run_rgb_to_grayscale, run_vector_add
from .kernel_launch_reference import rgb_image, seeded_float32, vector_add_inputs
from .kernel_launch_simulator import LaunchOrder, LaunchResult, ThreadContext, launch
from .matmul_model_kernels import X_ROWS, run_naive_matmul
from .matmul_model_reference import matmul_inputs
from .memory_access_tiling import MatmulShape
from .sim_profile_config import SimProfileVerificationError

SEMANTICS_SEED = 75
ORDER_SEED = 75
STENCIL_N = 70
STENCIL_THREADS_PER_BLOCK = 32
MATMUL_SHAPE = MatmulShape(m=9, n=7, k=5)
CASES = ("vector-add-unguarded", "grayscale-unguarded", "stencil", "matmul-guarded")
# Taken on the base simulator, before round 1; see the module docstring.
SEMANTICS_SHA256 = "7b6d751094724ef06dca5adbeed4a4a0c38fbbf8a4b0a0cd7f497cc498456a24"
# The totals of those records. They follow from the digest, so a verifier
# pins them too: a report cannot keep the digest and claim fewer records.
SEMANTICS_TOTALS = {
    "runs": 8,
    "out_of_bounds_reads": 826,
    "out_of_bounds_writes": 308,
    "write_conflicts": 464,
    "read_write_conflicts": 140,
}


def stencil_kernel(t: ThreadContext, x: DeviceBuffer) -> None:
    """Run one unguarded thread of ``x[i] = x[i - 1] + x[i + 1]`` in place.

    Thread 0 reads index -1, the last threads read and write past the end,
    and each neighbor read races with that neighbor's own write.
    """

    i = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    x[i] = to_float32(x[i - 1] + x[i + 1])


def _orders() -> tuple[LaunchOrder, LaunchOrder]:
    return (LaunchOrder.canonical(), LaunchOrder.permuted(ORDER_SEED))


def _stencil(order: LaunchOrder) -> tuple[LaunchResult, str]:
    x = DeviceBuffer(
        "x", "float32", seeded_float32(SEMANTICS_SEED, "stencil", STENCIL_N)
    )
    result = launch(
        stencil_kernel, launch_1d(STENCIL_N, STENCIL_THREADS_PER_BLOCK), x, order=order
    )
    return result, x.sha256()


def _log_mapping(log: LaunchAccessLog) -> dict[str, object]:
    return {
        "counts": log.counts(),
        "writer_threads": dict(sorted(log.writer_threads.items())),
        "out_of_bounds": [record.to_mapping() for record in log.out_of_bounds],
        "canonical_out_of_bounds": [
            record.to_mapping() for record in log.canonical_out_of_bounds
        ],
        "write_conflicts": [item.to_mapping() for item in log.write_conflicts],
        "read_write_conflicts": [
            item.to_mapping() for item in log.read_write_conflicts
        ],
    }


def _run_mapping(
    case: str, order: LaunchOrder, result: LaunchResult, output_sha256: str
) -> dict[str, object]:
    return {
        "case": case,
        "order": order.to_mapping(),
        "threads_run": result.threads_run,
        "order_sha256": result.order_sha256,
        "output_sha256": output_sha256,
        "log": _log_mapping(result.log),
    }


def _kernel_run(case: str, order: LaunchOrder) -> KernelRun:
    if case == CASES[0]:
        a, b = vector_add_inputs(SEMANTICS_SEED, 1000)
        return run_vector_add(a, b, threads_per_block=256, order=order, guarded=False)
    if case == CASES[1]:
        rgb = rgb_image(SEMANTICS_SEED, 40, 30)
        return run_rgb_to_grayscale(
            rgb, width=40, height=30, block=(16, 16), order=order, guarded=False
        )
    a, b = matmul_inputs(SEMANTICS_SEED, MATMUL_SHAPE)
    return run_naive_matmul(
        a, b, MATMUL_SHAPE, block=(4, 4), mapping=X_ROWS, order=order
    )


def semantic_runs() -> list[dict[str, object]]:
    """Run every case in both orders; return each run with its full log."""

    runs = []
    for case in CASES:
        for order in _orders():
            if case == "stencil":
                result, digest = _stencil(order)
            else:
                run = _kernel_run(case, order)
                result, digest = run.result, run.output_sha256
            runs.append(_run_mapping(case, order, result, digest))
    return runs


def semantics_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    """Digest the runs and total the misuse they record.

    ``canonical_records_order_free`` and ``guarded_output_order_free`` restate
    the two order rules of the Day 73 simulator for these cases.
    """

    pairs = [runs[index : index + 2] for index in range(0, len(runs), 2)]
    totals = {
        key: sum(run["log"]["counts"][key] for run in runs)
        for key in (
            "out_of_bounds_reads",
            "out_of_bounds_writes",
            "write_conflicts",
            "read_write_conflicts",
        )
    }
    return {
        "sha256": canonical_sha256(runs),
        "runs": len(runs),
        **totals,
        "canonical_records_order_free": all(
            first["log"]["canonical_out_of_bounds"]
            == second["log"]["canonical_out_of_bounds"]
            for first, second in pairs
        ),
        "guarded_output_order_free": (
            pairs[-1][0]["output_sha256"] == pairs[-1][1]["output_sha256"]
        ),
    }


def checked_semantics() -> dict[str, object]:
    """Return the summary, or refuse a simulator whose records changed."""

    summary = semantics_summary(semantic_runs())
    if summary["sha256"] != SEMANTICS_SHA256:
        raise SimProfileVerificationError(
            "the simulator's misuse records differ from the pinned digest "
            f"{SEMANTICS_SHA256}; an optimization changed a semantic"
        )
    return summary
