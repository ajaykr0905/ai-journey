"""Roofline classification of naive, tiled, and compulsory matmul traffic.

Each traffic model becomes a ``roofline.KernelWorkload`` whose bytes are its
modeled global-memory traffic, and ``roofline.analyze_roofline`` classifies it
against declared peak-compute and memory-bandwidth ceilings. The compulsory
workload from ``roofline.matmul_workload`` reads every operand element once
and writes C once: the traffic floor that perfect on-chip reuse approaches.
Lower-bound times are roofline arithmetic, not observed kernel runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_pattern import MemoryAccessError
from .memory_access_tiling import (
    MatmulShape,
    NaiveMatmulTraffic,
    TiledMatmulTraffic,
    matmul_element_bytes,
)
from .roofline import (
    KernelWorkload,
    RooflineEstimate,
    RooflineHardware,
    analyze_roofline,
    matmul_workload,
)

MAX_SWEEP_TILES = 8


def traffic_workload(
    traffic: NaiveMatmulTraffic | TiledMatmulTraffic,
) -> KernelWorkload:
    """Express modeled global traffic as a roofline kernel workload."""

    if isinstance(traffic, NaiveMatmulTraffic):
        kind = "naive_global_matmul"
    elif isinstance(traffic, TiledMatmulTraffic):
        kind = f"tiled_global_matmul_t{traffic.tile}"
    else:
        raise TypeError("traffic must be a naive or tiled matmul traffic model")
    return KernelWorkload(
        kind=kind,
        flops=traffic.flops,
        bytes_read=traffic.load_bytes,
        bytes_written=traffic.store_bytes,
    )


def compulsory_workload(shape: MatmulShape, dtype: str = "float32") -> KernelWorkload:
    """Reuse the roofline module's read-once, write-once matmul traffic."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    matmul_element_bytes(dtype)
    return matmul_workload(rows=shape.m, inner=shape.k, columns=shape.n, dtype=dtype)


def roofline_estimate_mapping(estimate: RooflineEstimate) -> dict[str, object]:
    """Return the JSON-safe fields of one roofline estimate."""

    if not isinstance(estimate, RooflineEstimate):
        raise TypeError("estimate must be a RooflineEstimate")
    workload = estimate.workload
    return {
        "kind": workload.kind,
        "flops": workload.flops,
        "bytes_read": workload.bytes_read,
        "bytes_written": workload.bytes_written,
        "total_bytes": workload.total_bytes,
        "arithmetic_intensity": workload.arithmetic_intensity,
        "compute_seconds": estimate.compute_seconds,
        "memory_seconds": estimate.memory_seconds,
        "lower_bound_seconds": estimate.lower_bound_seconds,
        "attainable_flops_per_second": estimate.attainable_flops_per_second,
        "bottleneck": estimate.bottleneck,
    }


@dataclass(frozen=True)
class TileRoofline:
    """One tile width's traffic and its roofline classification."""

    traffic: TiledMatmulTraffic
    estimate: RooflineEstimate


@dataclass(frozen=True)
class TileRooflineSweep:
    """Compulsory, naive, and tiled traffic for one shape under one roofline."""

    shape: MatmulShape
    dtype: str
    hardware: RooflineHardware
    compulsory: RooflineEstimate
    naive: RooflineEstimate
    tiles: tuple[TileRoofline, ...]

    @property
    def bottleneck_counts(self) -> dict[str, int]:
        counts = {"compute": 0, "memory": 0, "ridge": 0}
        for item in self.tiles:
            counts[item.estimate.bottleneck] += 1
        return counts

    @property
    def first_compute_bound_tile(self) -> int | None:
        """Return the smallest swept tile that is compute-bound, if any."""

        for item in self.tiles:
            if item.estimate.bottleneck == "compute":
                return item.traffic.tile
        return None


def sweep_tile_rooflines(
    shape: MatmulShape,
    *,
    tiles: tuple[int, ...],
    hardware: RooflineHardware,
    dtype: str = "float32",
) -> TileRooflineSweep:
    """Classify naive traffic and every declared tile width on one roofline."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    if not isinstance(hardware, RooflineHardware):
        raise TypeError("hardware must be RooflineHardware")
    if type(tiles) is not tuple or not 1 <= len(tiles) <= MAX_SWEEP_TILES:
        raise MemoryAccessError(
            f"tiles must be a tuple of 1 to {MAX_SWEEP_TILES} tile widths"
        )
    if any(type(tile) is not int for tile in tiles):
        raise TypeError("every tile must be an integer")
    if list(tiles) != sorted(set(tiles)):
        raise MemoryAccessError("tiles must be unique and increasing")
    swept = []
    for tile in tiles:
        traffic = TiledMatmulTraffic(shape, tile=tile, dtype=dtype)
        swept.append(
            TileRoofline(traffic, analyze_roofline(traffic_workload(traffic), hardware))
        )
    return TileRooflineSweep(
        shape=shape,
        dtype=dtype,
        hardware=hardware,
        compulsory=analyze_roofline(compulsory_workload(shape, dtype), hardware),
        naive=analyze_roofline(
            traffic_workload(NaiveMatmulTraffic(shape, dtype)), hardware
        ),
        tiles=tuple(swept),
    )
