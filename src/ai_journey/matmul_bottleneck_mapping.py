"""Why x over rows runs worse than x over columns, from the Day 74 warp model.

Timings alone can show that one mapping is slower, but not why. The Day 74
model lab gives the reason: ``matmul_model_access`` lists the addresses that
the 32 lanes of one warp touch at each loop step and counts the 32-byte
sectors of each request. This module runs that model on the first warp of
the first threadgroup of each mapping at each sweep shape. It applies to the
Metal kernels only under the matmul-metal lab's assumption: a SIMD group of
width 32 holds 32 consecutive x-threads of one threadgroup row. The module
refuses any other SIMD width.

The model counts every request as if no earlier request had brought its
sectors on chip, so it predicts the direction of the difference, not its
size. A verdict claims that x over rows is worse only when both hold:

* the model gives the x-over-rows warp more sectors than the x-over-columns
  warp, so there is a reason;
* the Day 74 median rate of x over columns is at least the pinned ratio above
  that of x over rows, so the timings show the effect.

Any other combination claims nothing and says which half is missing.
"""

from __future__ import annotations

from .kernel_launch_config import Index3
from .matmul_bottleneck_config import MappingRule, MatmulBottleneckError
from .matmul_metal_mapping import X_COLUMNS, X_ROWS
from .matmul_metal_source import KERNEL_FOR_MAPPING
from .matmul_model_access import (
    MatmulLaunch,
    WarpRef,
    operand_traffic,
    request_mapping,
    warp_requests,
)
from .matmul_model_reference import MatmulModelError
from .memory_access_pattern import WARP_SIZE
from .memory_access_tiling import MatmulShape

WORSE_AS_MODELED = "x_rows_worse_as_modeled"
NOT_SHOWN_BY_TIMING = "model_predicts_x_rows_worse_but_timing_does_not_show_it"
NO_MODEL_REASON = "x_rows_slower_without_a_model_reason"
NO_DIFFERENCE = "no_difference_in_model_or_timing"
VERDICTS = (WORSE_AS_MODELED, NOT_SHOWN_BY_TIMING, NO_MODEL_REASON, NO_DIFFERENCE)
PROBE_WARP = {"block": [0, 0, 0], "warp": 0}
MODEL_SCOPE = (
    "The Day 74 per-warp model of the first warp of threadgroup (0, 0): lane "
    "addresses from the kernels' index formulas, 32-byte sectors per request, "
    "no cache. It assumes that a SIMD group holds 32 consecutive x-threads. "
    "It predicts the direction of a difference, not its size."
)


def _first_request(requests: tuple) -> dict[str, object]:
    details = request_mapping(requests[0])
    del details["lane_addresses"]
    return details


def warp_model(
    shape: MatmulShape, group_size: tuple[int, int], mapping: str
) -> dict[str, object]:
    """Count the requests, sectors and lines of one probe warp of one mapping."""

    try:
        launch = MatmulLaunch(shape, tuple(group_size), mapping)
    except MatmulModelError as error:
        raise MatmulBottleneckError(f"warp model for {mapping}: {error}") from error
    requests = warp_requests(launch, WarpRef(Index3(0, 0), 0))
    if requests is None:
        # Lane 0 of the probe warp computes C[0, 0], so this cannot happen.
        raise MatmulBottleneckError(f"the probe warp of {mapping} is idle")
    operands = {
        "a": requests.a_loads,
        "b": requests.b_loads,
        "c": (requests.c_store,),
    }
    traffic = {name: operand_traffic(items) for name, items in operands.items()}
    return {
        "mapping": mapping,
        "active_lanes": requests.active_lanes,
        "operands": {
            name: {
                **traffic[name].to_mapping(),
                "first_request": _first_request(operands[name]),
            }
            for name in operands
        },
        "sectors": sum(item.sectors for item in traffic.values()),
        "lines": sum(item.lines for item in traffic.values()),
    }


def _verdict(more_sectors: bool, slower: bool) -> str:
    if more_sectors and slower:
        return WORSE_AS_MODELED
    if more_sectors:
        return NOT_SHOWN_BY_TIMING
    if slower:
        return NO_MODEL_REASON
    return NO_DIFFERENCE


def compare_mappings(
    shape: MatmulShape,
    group_size: tuple[int, int],
    simd_width: int,
    rates: dict[str, float],
    rule: MappingRule,
) -> dict[str, object]:
    """Join the warp model of both mappings with their Day 74 median rates.

    ``rates`` maps each naive kernel name to its median GFLOP/s.
    """

    if simd_width != WARP_SIZE:
        raise MatmulBottleneckError(
            f"the warp model needs a SIMD width of {WARP_SIZE}, not {simd_width}"
        )
    if type(rule) is not MappingRule:
        raise TypeError("rule must be a MappingRule")
    models = {
        mapping: warp_model(shape, group_size, mapping)
        for mapping in (X_COLUMNS, X_ROWS)
    }
    columns_rate = rates[KERNEL_FOR_MAPPING[X_COLUMNS]]
    rows_rate = rates[KERNEL_FOR_MAPPING[X_ROWS]]
    sector_ratio = models[X_ROWS]["sectors"] / models[X_COLUMNS]["sectors"]
    rate_ratio = columns_rate / rows_rate
    more_sectors = models[X_ROWS]["sectors"] > models[X_COLUMNS]["sectors"]
    slower = rate_ratio >= rule.slower_ratio_min
    return {
        "probe_warp": dict(PROBE_WARP),
        "warps": models,
        "model_sector_ratio": sector_ratio,
        "model_line_ratio": models[X_ROWS]["lines"] / models[X_COLUMNS]["lines"],
        "model_predicts_x_rows_worse": more_sectors,
        "x_columns_to_x_rows_rate_ratio": rate_ratio,
        "timing_shows_x_rows_slower": slower,
        "verdict": _verdict(more_sectors, slower),
    }
