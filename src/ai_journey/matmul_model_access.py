"""Per-warp global-memory requests of the naive matmul kernels.

CUDA forms each warp from 32 consecutive linear thread IDs of one block, x
fastest (CUDA C++ Programming Guide v12.6, section 4.1). This model requires
``blockDim.x`` to be a multiple of 32, so each warp is 32 consecutive
x-threads that share one ``threadIdx.y``. In loop step ``i`` a warp issues one
load from A and one load from B, and after the loop it issues one store to C.
A request holds the byte address of each of the 32 lanes, with ``None`` for a
lane that the guard turns off. A warp whose lanes are all off issues nothing.

Each matrix starts at byte 0 of its own allocation, and the runtime aligns
every allocation to at least 256 bytes (Programming Guide section 5.3.2), so
sector and line numbers count per matrix. A 4-byte element never crosses a
32-byte sector or a 128-byte line. Each request gets one class:

* ``single``: one lane is active, so the request reads or writes one element;
* ``broadcast``: two or more active lanes read one address;
* ``coalesced``: each active lane reads or writes the element at byte
  ``base + 4 * lane`` for one ``base``;
* ``strided``: any other pattern.

A request with one active lane has one address and no lane step, so it could
count as a broadcast or as a coalesced request, and a C store from one lane is
not a broadcast. Such a request has its own class. The guard leaves one lane
in an edge warp when the axis that x covers, ``n`` or ``m``, is one more than
a multiple of 32.

No cache is modeled. Each request counts its sectors as if no earlier request
had brought them on chip, as the memory_access lab's coalescing model does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernel_launch_config import Index3, LaunchConfig
from .matmul_model_kernels import X_COLUMNS, matmul_launch, require_mapping
from .matmul_model_reference import MatmulModelError, require_shape
from .memory_access_pattern import LINE_BYTES, SECTOR_BYTES, WARP_SIZE, WarpAccess
from .memory_access_tiling import MatmulShape

ELEMENT_BYTES = 4
OPERANDS = ("a", "b", "c")
ACCESS_CLASSES = ("single", "broadcast", "coalesced", "strided")


@dataclass(frozen=True)
class MatmulLaunch:
    """One naive matmul launch: a shape, a 2D block and a thread mapping."""

    shape: MatmulShape
    block: tuple[int, int]
    mapping: str

    def __post_init__(self) -> None:
        require_shape(self.shape)
        require_mapping(self.mapping)
        matmul_launch(self.shape, self.block, self.mapping)
        if self.block[0] % WARP_SIZE:
            raise MatmulModelError(
                f"the warp model needs block_x to be a multiple of {WARP_SIZE}"
            )

    @property
    def config(self) -> LaunchConfig:
        return matmul_launch(self.shape, self.block, self.mapping)

    @property
    def warps_per_block(self) -> int:
        return self.block[0] * self.block[1] // WARP_SIZE


@dataclass(frozen=True)
class WarpRef:
    """One warp of a launch: its block and its index among that block's warps."""

    block: Index3
    warp: int

    def to_mapping(self) -> dict[str, object]:
        return {"block": self.block.to_list(), "warp": self.warp}


def _require_launch(launch: object) -> MatmulLaunch:
    if type(launch) is not MatmulLaunch:
        raise TypeError("launch must be MatmulLaunch")
    return launch


def warp_refs(launch: MatmulLaunch) -> tuple[WarpRef, ...]:
    """Return every warp of a launch, blocks by linear ID, x fastest."""

    config = _require_launch(launch).config
    return tuple(
        WarpRef(Index3(block_x, block_y), warp)
        for block_y in range(config.grid.y)
        for block_x in range(config.grid.x)
        for warp in range(launch.warps_per_block)
    )


def lane_elements(launch: MatmulLaunch, ref: WarpRef) -> tuple[tuple[int, int], ...]:
    """Return the ``(row, col)`` of C that each lane of one warp computes."""

    config = _require_launch(launch).config
    if type(ref) is not WarpRef:
        raise TypeError("ref must be WarpRef")
    if (
        ref.block.x >= config.grid.x
        or ref.block.y >= config.grid.y
        or ref.block.z != 0
        or type(ref.warp) is not int
        or not 0 <= ref.warp < launch.warps_per_block
    ):
        raise MatmulModelError("the warp is outside the launch")
    block_x, block_y = launch.block
    thread_y, first_x = divmod(ref.warp * WARP_SIZE, block_x)
    y = ref.block.y * block_y + thread_y
    lanes = []
    for lane in range(WARP_SIZE):
        x = ref.block.x * block_x + first_x + lane
        lanes.append((y, x) if launch.mapping == X_COLUMNS else (x, y))
    return tuple(lanes)


def warp_of_element(launch: MatmulLaunch, row: int, col: int) -> WarpRef:
    """Return the warp whose lane computes ``C[row, col]``."""

    shape = _require_launch(launch).shape
    for value, size in ((row, shape.m), (col, shape.n)):
        if type(value) is not int:
            raise TypeError("row and col must be integers")
        if not 0 <= value < size:
            raise MatmulModelError("the element is outside C")
    x, y = (col, row) if launch.mapping == X_COLUMNS else (row, col)
    block_x, block_y = launch.block
    local = x % block_x + block_x * (y % block_y)
    return WarpRef(Index3(x // block_x, y // block_y), local // WARP_SIZE)


@dataclass(frozen=True)
class WarpRequests:
    """The loads of every loop step and the one store of one warp."""

    ref: WarpRef
    active_lanes: int
    a_loads: tuple[WarpAccess, ...]
    b_loads: tuple[WarpAccess, ...]
    c_store: WarpAccess


def warp_requests(launch: MatmulLaunch, ref: WarpRef) -> WarpRequests | None:
    """Return the requests of one warp, or ``None`` if the guard idles it.

    Lane addresses follow the kernel's index expressions: ``A[row, i]`` at
    ``(row * k + i) * 4``, ``B[i, col]`` at ``(i * n + col) * 4`` and
    ``C[row, col]`` at ``(row * n + col) * 4``.
    """

    shape = launch.shape
    m, n, k = shape.m, shape.n, shape.k
    lanes = [
        (row, col) if row < m and col < n else None
        for row, col in lane_elements(launch, ref)
    ]
    active = sum(lane is not None for lane in lanes)
    if not active:
        return None

    def request(element: int, row_stride: int, col_stride: int) -> WarpAccess:
        return WarpAccess(
            ELEMENT_BYTES,
            tuple(
                None
                if lane is None
                else (element + lane[0] * row_stride + lane[1] * col_stride)
                * ELEMENT_BYTES
                for lane in lanes
            ),
        )

    return WarpRequests(
        ref=ref,
        active_lanes=active,
        a_loads=tuple(request(i, k, 0) for i in range(k)),
        b_loads=tuple(request(i * n, 0, 1) for i in range(k)),
        c_store=request(0, n, 1),
    )


def sector_indices(access: WarpAccess) -> tuple[int, ...]:
    """Return the distinct 32-byte sectors that a request touches."""

    return tuple(
        sorted({address // SECTOR_BYTES for address in access.active_addresses})
    )


def line_indices(access: WarpAccess) -> tuple[int, ...]:
    """Return the distinct 128-byte lines that a request touches."""

    return tuple(sorted({address // LINE_BYTES for address in access.active_addresses}))


def lane_stride_bytes(access: WarpAccess) -> int | None:
    """Return the byte step per lane if the addresses are affine in the lane.

    A request with one active lane has no step, and neither has a pattern
    such as a permutation that no single step describes.
    """

    lanes, addresses = access.active_lanes, access.active_addresses
    if len(lanes) < 2:
        return None
    step, rest = divmod(addresses[1] - addresses[0], lanes[1] - lanes[0])
    if rest or any(
        address != addresses[0] + (lane - lanes[0]) * step
        for lane, address in zip(lanes, addresses)
    ):
        return None
    return step


def access_class(access: WarpAccess) -> str:
    """Classify one request as single, broadcast, coalesced or strided."""

    if len(access.active_lanes) == 1:
        return "single"
    if len(set(access.active_addresses)) == 1:
        return "broadcast"
    if lane_stride_bytes(access) == access.access_bytes:
        return "coalesced"
    return "strided"


def request_mapping(access: WarpAccess) -> dict[str, object]:
    """Return the lane addresses of one request and what the model counts."""

    return {
        "lane_addresses": list(access.lane_addresses),
        "active_lanes": len(access.active_lanes),
        "distinct_addresses": len(set(access.active_addresses)),
        "sectors": len(sector_indices(access)),
        "lines": len(line_indices(access)),
        "class": access_class(access),
        "lane_stride_bytes": lane_stride_bytes(access),
    }


@dataclass(frozen=True)
class OperandTraffic:
    """Totals of the requests to one matrix over every warp of a launch.

    ``requested_bytes`` counts 4 bytes for each active lane, and
    ``unique_bytes`` counts each distinct address of a request once. Both add
    up request by request, with no reuse between requests.
    """

    requests: int
    sectors: int
    lines: int
    requested_bytes: int
    unique_bytes: int
    classes: tuple[tuple[str, int], ...]

    @property
    def transferred_bytes(self) -> int:
        return self.sectors * SECTOR_BYTES

    def to_mapping(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "sectors": self.sectors,
            "lines": self.lines,
            "requested_bytes": self.requested_bytes,
            "unique_bytes": self.unique_bytes,
            "transferred_bytes": self.transferred_bytes,
            "classes": dict(self.classes),
        }


def operand_traffic(requests: tuple[WarpAccess, ...]) -> OperandTraffic:
    """Add up the requests to one matrix without assuming any cache."""

    classes = dict.fromkeys(ACCESS_CLASSES, 0)
    for access in requests:
        classes[access_class(access)] += 1
    return OperandTraffic(
        requests=len(requests),
        sectors=sum(len(sector_indices(access)) for access in requests),
        lines=sum(len(line_indices(access)) for access in requests),
        requested_bytes=sum(
            len(access.active_lanes) * access.access_bytes for access in requests
        ),
        unique_bytes=sum(
            len(set(access.active_addresses)) * access.access_bytes
            for access in requests
        ),
        classes=tuple(classes.items()),
    )


@dataclass(frozen=True)
class LaunchAccess:
    """Warp counts and per-matrix request totals of one whole launch."""

    launch: MatmulLaunch
    warps: int
    full_warps: int
    partial_warps: int
    idle_warps: int
    operands: tuple[tuple[str, OperandTraffic], ...]

    @property
    def transferred_bytes(self) -> int:
        return sum(traffic.transferred_bytes for _, traffic in self.operands)

    def to_mapping(self) -> dict[str, object]:
        return {
            "warps": {
                "launched": self.warps,
                "full": self.full_warps,
                "partial": self.partial_warps,
                "idle": self.idle_warps,
            },
            "operands": {name: traffic.to_mapping() for name, traffic in self.operands},
            "transferred_bytes": self.transferred_bytes,
        }


def analyze_launch(launch: MatmulLaunch) -> LaunchAccess:
    """Enumerate every warp of a launch and total its requests per matrix."""

    refs = warp_refs(launch)
    loads: dict[str, list[WarpAccess]] = {name: [] for name in OPERANDS}
    full = partial = idle = 0
    for ref in refs:
        requests = warp_requests(launch, ref)
        if requests is None:
            idle += 1
            continue
        if requests.active_lanes == WARP_SIZE:
            full += 1
        else:
            partial += 1
        loads["a"].extend(requests.a_loads)
        loads["b"].extend(requests.b_loads)
        loads["c"].append(requests.c_store)
    return LaunchAccess(
        launch=launch,
        warps=len(refs),
        full_warps=full,
        partial_warps=partial,
        idle_warps=idle,
        operands=tuple(
            (name, operand_traffic(tuple(loads[name]))) for name in OPERANDS
        ),
    )
