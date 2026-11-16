"""Deterministic launch simulator for per-thread Python kernels.

``launch`` calls ``kernel(ctx, *args)`` once for every (blockIdx, threadIdx)
pair of a validated ``LaunchConfig``. The context holds the CUDA built-ins
``blockIdx``, ``threadIdx``, ``blockDim`` and ``gridDim`` as ``block_idx``,
``thread_idx``, ``block_dim`` and ``grid_dim``. Threads run one at a time in
a declared order: the canonical order, or a seeded permutation of all the
threads of the grid. CUDA does not order the threads of a launch, so a
correct kernel gives the same result in every order. The permuted orders
test that claim, and the instrumented buffers report the races that break it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from .kernel_launch_buffer import AccessRecorder, DeviceBuffer, LaunchAccessLog
from .kernel_launch_config import (
    Dim3,
    Index3,
    KernelLaunchError,
    LaunchConfig,
    require_int,
)

CANONICAL = "canonical"
PERMUTED = "permuted"
MAX_SIMULATED_THREADS = 1 << 20
MAX_ORDER_SEED = 2**63 - 1
_ORDER_LABEL = "ai-journey-kernel-launch-order"


@dataclass(frozen=True)
class ThreadContext:
    """The CUDA built-in variables that one simulated thread reads."""

    block_idx: Index3
    thread_idx: Index3
    block_dim: Dim3
    grid_dim: Dim3

    def __post_init__(self) -> None:
        if (
            type(self.block_idx) is not Index3
            or type(self.thread_idx) is not Index3
            or type(self.block_dim) is not Dim3
            or type(self.grid_dim) is not Dim3
        ):
            raise TypeError("a thread context holds two Index3 and two Dim3 values")


@dataclass(frozen=True)
class LaunchOrder:
    """The order in which the simulator runs threads: canonical or seeded.

    CUDA promises no thread order. The canonical order is only this
    simulator's declared default, and a seed names one reproducible
    permutation of every thread in the grid.
    """

    kind: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind == CANONICAL:
            if self.seed is not None:
                raise KernelLaunchError("the canonical order takes no seed")
        elif self.kind == PERMUTED:
            require_int(self.seed, "order seed", minimum=0, maximum=MAX_ORDER_SEED)
        else:
            raise KernelLaunchError(f"order kind must be {CANONICAL!r} or {PERMUTED!r}")

    @classmethod
    def canonical(cls) -> LaunchOrder:
        return cls(CANONICAL)

    @classmethod
    def permuted(cls, seed: int) -> LaunchOrder:
        return cls(PERMUTED, seed)

    def to_mapping(self) -> dict[str, object]:
        return {"kind": self.kind, "seed": self.seed}

    @classmethod
    def from_mapping(cls, mapping: object) -> LaunchOrder:
        """Parse ``{"kind": ..., "seed": ...}`` with no missing or extra field."""

        if not isinstance(mapping, Mapping) or set(mapping) != {"kind", "seed"}:
            raise KernelLaunchError("an order must hold exactly kind and seed")
        return cls(mapping["kind"], mapping["seed"])


def _uniform_below(seed: int, counter: int, bound: int) -> int:
    limit = (1 << 64) - (1 << 64) % bound
    attempt = 0
    while True:
        digest = sha256(
            f"{_ORDER_LABEL}:{seed}:{counter}:{attempt}".encode("ascii")
        ).digest()
        value = int.from_bytes(digest[:8], "big")
        if value < limit:
            return value % bound
        attempt += 1


def thread_order(config: LaunchConfig, order: LaunchOrder) -> tuple[int, ...]:
    """Return the linear thread IDs of a launch in the order they run.

    The canonical order runs blocks by linear block ID and the threads of a
    block by linear thread ID, both x fastest, which is ascending linear ID.
    A permuted order shuffles every thread of the grid with Fisher-Yates.
    SHA-256 counter blocks drive the shuffle instead of ``random``, so one
    seed gives the same order on every platform and Python version.
    """

    if type(config) is not LaunchConfig:
        raise TypeError("config must be LaunchConfig")
    if type(order) is not LaunchOrder:
        raise TypeError("order must be LaunchOrder")
    count = config.launched_threads
    if count > MAX_SIMULATED_THREADS:
        raise KernelLaunchError(
            f"the launch has {count} threads; the simulator runs at most "
            f"{MAX_SIMULATED_THREADS}"
        )
    thread_ids = list(range(count))
    if order.kind == PERMUTED:
        for index in range(count - 1, 0, -1):
            other = _uniform_below(order.seed, index, index + 1)
            thread_ids[index], thread_ids[other] = thread_ids[other], thread_ids[index]
    return tuple(thread_ids)


def order_sha256(thread_ids: Sequence[int]) -> str:
    """Digest a run order as 8-byte big-endian linear thread IDs."""

    return sha256(
        b"".join(thread_id.to_bytes(8, "big") for thread_id in thread_ids)
    ).hexdigest()


@dataclass(frozen=True)
class LaunchResult:
    """What one simulated launch ran, in which order, and what memory saw."""

    config: LaunchConfig
    order: LaunchOrder
    threads_run: int
    order_sha256: str
    log: LaunchAccessLog


def launch(
    kernel: Callable[..., None],
    config: LaunchConfig,
    *args: object,
    order: LaunchOrder | None = None,
) -> LaunchResult:
    """Run ``kernel(ctx, *args)`` once per thread of ``config`` in ``order``.

    Arguments are device buffers or int and float scalars, as a CUDA launch
    passes pointers and values. A kernel returns nothing, like a
    ``__global__`` function. Other arguments are refused because the kernel
    could reach memory through them that no recorder sees.
    """

    if not callable(kernel):
        raise TypeError("kernel must be callable")
    if type(config) is not LaunchConfig:
        raise TypeError("config must be LaunchConfig")
    order = LaunchOrder.canonical() if order is None else order
    for argument in args:
        if type(argument) not in (DeviceBuffer, int, float):
            raise TypeError(
                "kernel arguments must be DeviceBuffer values or int or float scalars"
            )
    thread_ids = thread_order(config, order)
    per_block = config.threads_per_block
    blocks = [
        config.coordinates(block_id * per_block)[0] for block_id in range(config.blocks)
    ]
    threads = [config.coordinates(local)[1] for local in range(per_block)]
    recorder = AccessRecorder(
        config, [argument for argument in args if type(argument) is DeviceBuffer]
    )
    try:
        for thread_id in thread_ids:
            block_id, local = divmod(thread_id, per_block)
            context = ThreadContext(
                blocks[block_id], threads[local], config.block, config.grid
            )
            recorder.enter(thread_id)
            try:
                returned = kernel(context, *args)
            finally:
                recorder.leave()
            if returned is not None:
                raise KernelLaunchError(
                    "a kernel returns nothing, like a CUDA __global__ function"
                )
    finally:
        log = recorder.finish()
    return LaunchResult(
        config=config,
        order=order,
        threads_run=len(thread_ids),
        order_sha256=order_sha256(thread_ids),
        log=log,
    )
