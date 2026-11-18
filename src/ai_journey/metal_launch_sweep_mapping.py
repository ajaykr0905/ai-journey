"""Check the matmul's x-over-columns mapping for every shape in the Day 73 simulator.

A GPU run shows only the outputs. It cannot show a padding thread that read
past the end of ``A`` or ``B``, or two threads that wrote one element. The Day
73 launch simulator runs ``matmul_thread`` once for every thread of a whole
threadgroup grid, attributes each buffer access to its thread, and records
out-of-bounds accesses and write conflicts. Each admitted shape runs the
pinned ``MAPPING_CASE`` in the canonical order and in one seeded permuted
order, because CUDA and Metal promise no order between threads.

``MAPPING_CASE`` has 13 rows and 19 columns, both prime, so a shape with two
or more threads on an axis leaves padding threads on that axis. The inputs
are small integers, so every product and partial sum is exact in float32,
and each order must reproduce the exact integer product bit for bit. A shape
passes when it launches whole threadgroups, writes each element of ``C`` once
from one thread, makes ``2 m n k`` in-bounds reads and no out-of-bounds
access, has no conflict, and gives the exact product in both orders.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernel_launch_buffer import DeviceBuffer, values_sha256
from .kernel_launch_config import launch_2d
from .kernel_launch_simulator import MAX_ORDER_SEED, LaunchOrder, launch
from .kernel_metal_geometry import require_integer
from .metal_launch_sweep_shapes import MetalLaunchSweepError, group_shape
from .metal_launch_sweep_source import matmul_thread

# m rows, n columns and the inner size k of the simulated product.
MAPPING_CASE = (13, 19, 3)


def mapping_inputs() -> tuple[list[float], list[float]]:
    """Return row-major ``A`` and ``B`` of ``MAPPING_CASE`` as small integers."""

    m, n, k = MAPPING_CASE
    a = [float((row + 2 * inner) % 5 - 2) for row in range(m) for inner in range(k)]
    b = [float((3 * inner + col) % 7 - 3) for inner in range(k) for col in range(n)]
    return a, b


def exact_product() -> list[float]:
    """Return ``C = A B`` of ``MAPPING_CASE`` in exact integer arithmetic."""

    m, n, k = MAPPING_CASE
    a, b = mapping_inputs()
    return [
        float(sum(a[row * k + inner] * b[inner * n + col] for inner in range(k)))
        for row in range(m)
        for col in range(n)
    ]


@dataclass(frozen=True)
class MappingCheck:
    """What the simulator saw one shape do to memory, in both thread orders."""

    group_size: tuple[int, int]
    order_seed: int
    launched_threads: int
    writer_threads: int
    reads: int
    writes: int
    out_of_bounds: int
    write_conflicts: int
    read_write_conflicts: int
    output_sha256: str
    exact: bool

    def to_mapping(self) -> dict[str, object]:
        """Return the counts a report records and a verifier can rederive."""

        return {
            "group_size": list(self.group_size),
            "order_seed": self.order_seed,
            "launched_threads": self.launched_threads,
            "writer_threads": self.writer_threads,
            "reads": self.reads,
            "writes": self.writes,
            "out_of_bounds": self.out_of_bounds,
            "write_conflicts": self.write_conflicts,
            "read_write_conflicts": self.read_write_conflicts,
            "output_sha256": self.output_sha256,
            "exact": self.exact,
        }


def _run(group: tuple[int, int], order: LaunchOrder) -> tuple[object, list[float]]:
    m, n, k = MAPPING_CASE
    a, b = mapping_inputs()
    c = DeviceBuffer.zeros("c", "float32", m * n)
    config = launch_2d(n, m, *group)
    result = launch(
        matmul_thread,
        config,
        DeviceBuffer("a", "float32", a),
        DeviceBuffer("b", "float32", b),
        c,
        m,
        n,
        k,
        order=order,
    )
    return result, list(c.to_host())


def check_mapping(group: tuple[int, int], *, order_seed: int) -> MappingCheck:
    """Simulate one shape in both orders and require the mapping to be clean.

    A shape that breaks the mapping stops the run here, before any GPU
    launch, so a report never holds a failed mapping check.
    """

    shape = group_shape(group, 2)
    seed = require_integer(order_seed, "order_seed", 0, MAX_ORDER_SEED)
    m, n, k = MAPPING_CASE
    expected = exact_product()
    canonical, first = _run(shape, LaunchOrder.canonical())
    permuted, second = _run(shape, LaunchOrder.permuted(seed))
    log = canonical.log
    counts = log.counts()
    clean = (
        log.memory_safe
        and log.race_free
        and permuted.log.memory_safe
        and permuted.log.race_free
        and counts == permuted.log.counts()
        and log.writer_threads["c"] == m * n
        and counts["writes"] == m * n
        and counts["reads"] == 2 * m * n * k
    )
    exact = first == expected and second == expected
    if not clean or not exact:
        raise MetalLaunchSweepError(
            f"the simulated matmul with group {list(shape)} broke its mapping: "
            f"{counts}, exact={exact}"
        )
    return MappingCheck(
        group_size=shape,
        order_seed=seed,
        launched_threads=canonical.threads_run,
        writer_threads=log.writer_threads["c"],
        reads=counts["reads"],
        writes=counts["writes"],
        out_of_bounds=counts["out_of_bounds_reads"] + counts["out_of_bounds_writes"],
        write_conflicts=counts["write_conflicts"],
        read_write_conflicts=counts["read_write_conflicts"],
        output_sha256=values_sha256("float32", first),
        exact=exact,
    )
