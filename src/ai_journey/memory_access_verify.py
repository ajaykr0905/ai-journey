"""Independent verification of sealed memory-access reports.

Analytical reports are checked twice. First, every analytical number is
recomputed here from first principles without the model modules: each
pattern's lane addresses from its declared parameters, sectors, lines and
banks by enumerating the bytes each lane touches, tile traffic by
clipping every partial block and phase, and roofline times from the declared
ceilings. Second, the whole report is rebuilt from its embedded plan and must
match byte for byte, which also pins names, sources and exclusions.

Measurement reports hold timings that cannot be recomputed. They get
structural and accounting checks instead: exact schemas, the plan binding,
the CPU-only evidence boundary, nearest-rank statistics recomputed from the
raw samples, FLOP and GFLOP/s arithmetic, correctness bounds, and summaries.
A matching digest shows internal consistency; it cannot prove that a timing
was observed.
"""

from __future__ import annotations

import math
import re
from hashlib import sha256

from .memory_access_benchmark import PERCENTILE_METHOD, TIMER
from .memory_access_pattern import MemoryAccessError
from .memory_access_plan import (
    BenchmarkSpec,
    MemoryAccessPlan,
    canonical_json_bytes,
    plan_from_mapping,
)
from .memory_access_report import (
    ANALYSIS_BOUNDARY,
    ANALYSIS_KIND,
    CORRECTNESS_SOURCE,
    DIGEST_FIELD,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_KIND,
    MEASUREMENT_SCOPE,
    build_analysis_report,
    report_digest,
    seal_report,
)
from .memory_access_tiling import MatmulShape

_LANES = 32
_SECTOR = 32
_LINE = 128
_WORD = 4
_BANKS = 32
_ELEMENT_BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_ANALYSIS_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "evidence_boundary",
    "plan",
    "plan_sha256",
    "model_sources",
    "exclusions",
    "patterns",
    "transposes",
    "matmul",
    "summary",
    DIGEST_FIELD,
}
_MEASUREMENT_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "evidence_boundary",
    "plan_name",
    "plan_sha256",
    "benchmark",
    "timer",
    "percentile_method",
    "correctness_source",
    "environment",
    "results",
    "summary",
    DIGEST_FIELD,
}
_RESULT_FIELDS = {
    "backend",
    "device",
    "dtype",
    "m",
    "n",
    "k",
    "warmup",
    "repeats",
    "seed",
    "input_sha256",
    "samples_ns",
    "timing",
    "flops",
    "median_gflops_per_second",
    "best_gflops_per_second",
    "correctness",
}
_ENVIRONMENT_FIELDS = {
    "python_version",
    "python_implementation",
    "numpy_version",
    "torch_version",
    "platform_system",
    "platform_machine",
    "cpu_count",
    "torch_threads",
    "torch_float32_matmul_precision",
}


class MemoryAccessVerificationError(MemoryAccessError):
    """Raised when a sealed report does not survive independent checks."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryAccessVerificationError(message)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _permutation(seed: int) -> list[int]:
    """Re-derive the SHA-256 counter Fisher-Yates order of one warp's lanes."""

    order = list(range(_LANES))
    for index in range(_LANES - 1, 0, -1):
        bound = index + 1
        attempt = 0
        while True:
            label = f"ai-journey-memory-access-permutation:{seed}:{index}:{attempt}"
            value = int.from_bytes(sha256(label.encode()).digest()[:8], "big")
            if value < 2**64 - 2**64 % bound:
                break
            attempt += 1
        other = value % bound
        order[index], order[other] = order[other], order[index]
    return order


def _declared_lanes(kind: str, values: dict[str, int]) -> list[int | None]:
    """Recompute one pattern's per-lane byte addresses from its parameters."""

    width = values["access_bytes"]
    base = values.get("base_bytes", 0)
    if kind == "broadcast":
        lanes = [values["address_bytes"]] * _LANES
    elif kind == "permutation":
        lanes = [base + element * width for element in _permutation(values["seed"])]
    elif kind == "row":
        first = values.get("start_column", 0)
        lanes = [
            base + (values["row"] * values["leading_dimension"] + column) * width
            if column < values["columns"]
            else None
            for column in range(first, first + _LANES)
        ]
    elif kind == "column":
        first = values.get("start_row", 0)
        lanes = [
            base + (row * values["leading_dimension"] + values["column"]) * width
            if row < values["rows"]
            else None
            for row in range(first, first + _LANES)
        ]
    else:
        stride = values.get("stride_elements", 1)
        lanes = [base + lane * stride * width for lane in range(_LANES)]
    mask = values.get("active_mask", (1 << _LANES) - 1)
    return [address if mask >> lane & 1 else None for lane, address in enumerate(lanes)]


def _global_metrics(lane_addresses: list, width: int) -> dict[str, object]:
    active = [address for address in lane_addresses if address is not None]
    touched = {byte for address in active for byte in range(address, address + width)}
    sectors = sorted({byte // _SECTOR for byte in touched})
    lines = sorted({byte // _LINE for byte in touched})
    transferred = len(sectors) * _SECTOR
    return {
        "access_bytes": width,
        "active_lanes": len(active),
        "sector_indices": sectors,
        "line_indices": lines,
        "sectors": len(sectors),
        "lines": len(lines),
        "requested_bytes": len(active) * width,
        "unique_bytes": len(touched),
        "transferred_bytes": transferred,
        "efficiency": len(touched) / transferred,
    }


def _bank_metrics(lane_addresses: list, width: int) -> dict[str, object]:
    active = [address for address in lane_addresses if address is not None]
    lanes_per_word: dict[int, int] = {}
    for address in active:
        for word in {byte // _WORD for byte in range(address, address + width)}:
            lanes_per_word[word] = lanes_per_word.get(word, 0) + 1
    banks: dict[int, set[int]] = {}
    for word in lanes_per_word:
        banks.setdefault(word % _BANKS, set()).add(word)
    degree = max(len(words) for words in banks.values())
    return {
        "access_bytes": width,
        "active_lanes": len(active),
        "distinct_words": len(lanes_per_word),
        "multicast_words": sum(1 for count in lanes_per_word.values() if count > 1),
        "banks_used": len(banks),
        "conflict_degree": degree,
        "wavefronts": degree,
    }


def _bank_summary(analyses: list[dict[str, object]]) -> dict[str, object]:
    wavefronts = sum(item["wavefronts"] for item in analyses)
    return {
        "requests": len(analyses),
        "wavefronts": wavefronts,
        "excess_wavefronts": wavefronts - len(analyses),
        "max_conflict_degree": max(item["conflict_degree"] for item in analyses),
        "wavefronts_per_request": wavefronts / len(analyses),
    }


def _scaled_global(request: dict[str, object], count: int) -> dict[str, object]:
    sectors = request["sectors"] * count
    unique = request["unique_bytes"] * count
    return {
        "requests": count,
        "sectors": sectors,
        "lines": request["lines"] * count,
        "requested_bytes": request["requested_bytes"] * count,
        "unique_bytes": unique,
        "transferred_bytes": sectors * _SECTOR,
        "sectors_per_request": sectors / count,
        "efficiency": unique / (sectors * _SECTOR),
    }


def _check_patterns(report: dict[str, object], plan: MemoryAccessPlan) -> None:
    for entry, spec in zip(report["patterns"], plan.patterns, strict=True):
        access = entry["access"]
        width = access["access_bytes"]
        lanes = access["lane_addresses"]
        _require(
            (entry["name"], entry["kind"], width)
            == (spec.name, spec.kind, dict(spec.parameters)["access_bytes"])
            and type(lanes) is list
            and len(lanes) == _LANES
            and all(value is None or type(value) is int for value in lanes),
            f"pattern {spec.name} does not record its declared warp request",
        )
        mask = sum(1 << lane for lane, value in enumerate(lanes) if value is not None)
        _require(
            mask != 0 and access["active_mask"] == mask,
            "pattern lane mask does not reconcile",
        )
        _require(
            entry["global_memory"] == _global_metrics(lanes, width),
            f"pattern {entry['name']} global accounting does not recompute",
        )
        expected_shared = _bank_metrics(lanes, width) if width <= _WORD else None
        _require(
            entry["shared_memory"] == expected_shared,
            f"pattern {entry['name']} bank accounting does not recompute",
        )


def _check_lanes(report: dict[str, object], plan: MemoryAccessPlan) -> None:
    """Catch a generator bug that the rebuilt report would faithfully repeat."""

    for entry, spec in zip(report["patterns"], plan.patterns, strict=True):
        _require(
            entry["access"]["lane_addresses"]
            == _declared_lanes(spec.kind, dict(spec.parameters)),
            f"pattern {spec.name} lane addresses do not follow its declared parameters",
        )


def _check_transposes(report: dict[str, object], plan: MemoryAccessPlan) -> None:
    for entry, spec in zip(report["transposes"], plan.transposes, strict=True):
        _require(
            type(entry) is dict and entry.get("name") == spec.name,
            f"transpose {spec.name} is missing or out of order",
        )
        ld = spec.leading_dimension
        width = spec.element_bytes
        size = spec.matrix_size
        rows = [
            _bank_metrics([(row * ld + lane) * width for lane in range(_LANES)], width)
            for row in range(_LANES)
        ]
        columns = [
            _bank_metrics(
                [(lane * ld + column) * width for lane in range(_LANES)], width
            )
            for column in range(_LANES)
        ]
        requests = size * size // _LANES
        row_request = _global_metrics([lane * width for lane in range(_LANES)], width)
        column_request = _global_metrics(
            [lane * size * width for lane in range(_LANES)], width
        )
        expected = {
            "name": entry["name"],
            "tile": _LANES,
            "leading_dimension": ld,
            "element_bytes": width,
            "matrix_size": size,
            "tile_bytes": _LANES * ld * width,
            "padding_bytes": _LANES * (ld - _LANES) * width,
            "tiles": (size // _LANES) ** 2,
            "shared_row_writes_per_tile": _bank_summary(rows),
            "shared_column_reads_per_tile": _bank_summary(columns),
            "naive_global_loads": _scaled_global(row_request, requests),
            "naive_global_stores": _scaled_global(column_request, requests),
            "tiled_global_loads": _scaled_global(row_request, requests),
            "tiled_global_stores": _scaled_global(row_request, requests),
        }
        _require(entry == expected, f"transpose {entry['name']} does not recompute")


def _inner_loop(tile: int, width: int) -> dict[str, object]:
    warps = _ceil_div(tile * tile, _LANES)
    offsets = {
        "a_row_major": lambda ty, tx, step: ty * tile + step,
        "b_row_major": lambda ty, tx, step: step * tile + tx,
        "b_column_major": lambda ty, tx, step: tx * tile + step,
        "b_column_major_padded": lambda ty, tx, step: tx * (tile + 1) + step,
    }
    layouts = {}
    for name, offset in offsets.items():
        analyses = []
        for warp in range(warps):
            for step in range(tile):
                lanes = []
                for lane in range(_LANES):
                    thread = warp * _LANES + lane
                    if thread < tile * tile:
                        ty, tx = divmod(thread, tile)
                        lanes.append(offset(ty, tx, step) * width)
                    else:
                        lanes.append(None)
                analyses.append(_bank_metrics(lanes, width))
        layouts[name] = {
            "all_steps": _bank_summary(analyses),
            "warp0_step0": analyses[0],
        }
    return {
        "tile": tile,
        "element_bytes": width,
        "warps": warps,
        "steps": tile,
        "layouts": layouts,
    }


def _estimate(
    kind: str, flops: int, read: int, written: int, peak: float, bandwidth: float
) -> dict[str, object]:
    total = read + written
    intensity = flops / total
    compute = flops / peak
    memory = total / bandwidth
    if compute > memory:
        bottleneck = "compute"
    elif memory > compute:
        bottleneck = "memory"
    else:
        bottleneck = "ridge"
    return {
        "kind": kind,
        "flops": flops,
        "bytes_read": read,
        "bytes_written": written,
        "total_bytes": total,
        "arithmetic_intensity": intensity,
        "compute_seconds": compute,
        "memory_seconds": memory,
        "lower_bound_seconds": max(compute, memory),
        "attainable_flops_per_second": min(peak, bandwidth * intensity),
        "bottleneck": bottleneck,
    }


def _clipped_total(extent: int, tile: int) -> int:
    return sum(
        min(tile, extent - start * tile) for start in range(_ceil_div(extent, tile))
    )


def _tiled_traffic(m: int, n: int, k: int, tile: int, dtype: str) -> dict[str, object]:
    width = _ELEMENT_BYTES[dtype]
    rows, columns, phases = (_ceil_div(value, tile) for value in (m, n, k))
    useful_a = columns * _clipped_total(m, tile) * _clipped_total(k, tile)
    useful_b = rows * _clipped_total(k, tile) * _clipped_total(n, tile)
    slots = 2 * tile * tile * phases * rows * columns
    flops = 2 * m * n * k
    load_bytes = (useful_a + useful_b) * width
    total = load_bytes + m * n * width
    return {
        "tile": tile,
        "dtype": dtype,
        "element_bytes": width,
        "block_rows": rows,
        "block_columns": columns,
        "blocks": rows * columns,
        "phases": phases,
        "threads_per_block": tile * tile,
        "shared_memory_bytes_per_block": 2 * tile * tile * width,
        "tile_slot_loads": slots,
        "useful_a_loads": useful_a,
        "useful_b_loads": useful_b,
        "useful_loads": useful_a + useful_b,
        "padded_slots": slots - useful_a - useful_b,
        "store_elements": m * n,
        "load_bytes": load_bytes,
        "store_bytes": m * n * width,
        "total_bytes": total,
        "flops": flops,
        "arithmetic_intensity": flops / total,
        "reuse_factor": 2 * m * n * k / (useful_a + useful_b),
    }


def _check_matmul(report: dict[str, object], plan: MemoryAccessPlan) -> None:
    matmul = report["matmul"]
    dtype = plan.matmul.dtype
    width = _ELEMENT_BYTES[dtype]
    peak = plan.roofline.peak_flops_per_second
    bandwidth = plan.roofline.memory_bandwidth_bytes_per_second
    _require(
        matmul["roofline"]["ridge_point_flops_per_byte"] == peak / bandwidth,
        "declared ridge point does not recompute",
    )
    expected_banks = (
        [_inner_loop(tile, width) for tile in plan.matmul.tiles]
        if width <= _WORD
        else None
    )
    _require(
        matmul["inner_loop_banks"] == expected_banks,
        "inner-loop bank analysis does not recompute",
    )
    totals = {"compute": 0, "memory": 0, "ridge": 0}
    for sweep, shape in zip(matmul["sweeps"], plan.matmul.shapes, strict=True):
        m, n, k = shape.m, shape.n, shape.k
        flops = 2 * m * n * k
        compulsory = _estimate(
            "matmul", flops, (m * k + k * n) * width, m * n * width, peak, bandwidth
        )
        naive_traffic = {
            "dtype": dtype,
            "element_bytes": width,
            "load_elements": 2 * m * n * k,
            "store_elements": m * n,
            "load_bytes": 2 * m * n * k * width,
            "store_bytes": m * n * width,
            "total_bytes": (2 * m * n * k + m * n) * width,
            "flops": flops,
            "arithmetic_intensity": flops / ((2 * m * n * k + m * n) * width),
        }
        naive = _estimate(
            "naive_global_matmul",
            flops,
            naive_traffic["load_bytes"],
            naive_traffic["store_bytes"],
            peak,
            bandwidth,
        )
        tiled = []
        counts = {"compute": 0, "memory": 0, "ridge": 0}
        first_compute = None
        for tile in plan.matmul.tiles:
            traffic = _tiled_traffic(m, n, k, tile, dtype)
            estimate = _estimate(
                f"tiled_global_matmul_t{tile}",
                flops,
                traffic["load_bytes"],
                traffic["store_bytes"],
                peak,
                bandwidth,
            )
            counts[estimate["bottleneck"]] += 1
            if first_compute is None and estimate["bottleneck"] == "compute":
                first_compute = tile
            tiled.append({"traffic": traffic, "roofline": estimate})
        for name, count in counts.items():
            totals[name] += count
        expected = {
            "shape": {"m": m, "n": n, "k": k},
            "compulsory": compulsory,
            "naive": {"traffic": naive_traffic, "roofline": naive},
            "tiled": tiled,
            "bottleneck_counts": counts,
            "first_compute_bound_tile": first_compute,
        }
        _require(sweep == expected, f"matmul sweep {m}x{n}x{k} does not recompute")
    summary = report["summary"]
    _require(
        summary["tiled_bottleneck_counts"] == totals
        and summary["tiled_estimates"] == sum(totals.values())
        and summary["patterns"] == len(report["patterns"])
        and summary["shared_memory_patterns"]
        == sum(1 for item in report["patterns"] if item["shared_memory"] is not None)
        and summary["transposes"] == len(report["transposes"]),
        "analysis summary does not reconcile",
    )


def _seal_and_contract(report: object, fields: set[str], kind: str) -> dict:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == fields, "report fields do not match schema")
    _require(
        type(report["schema_version"]) is int and report["schema_version"] == 1,
        "unsupported report schema",
    )
    _require(report["kind"] == kind, "unexpected report kind")
    _require(
        report[DIGEST_FIELD] == report_digest(report),
        "report digest does not match its content",
    )
    return report


def _verify_analysis(
    report: object, plan: MemoryAccessPlan | None
) -> dict[str, object]:
    report = _seal_and_contract(report, _ANALYSIS_FIELDS, ANALYSIS_KIND)
    _require(report["measured"] is False, "analytical report must remain unmeasured")
    _require(
        report["evidence_boundary"] == ANALYSIS_BOUNDARY,
        "analytical evidence boundary was weakened",
    )
    embedded = plan_from_mapping(report["plan"])
    _require(report["plan_sha256"] == embedded.fingerprint(), "plan digest mismatch")
    if plan is not None:
        _require(
            embedded.fingerprint() == plan.fingerprint(),
            "report was not built from the supplied plan",
        )
    _require(
        len(report["patterns"]) == len(embedded.patterns)
        and len(report["transposes"]) == len(embedded.transposes)
        and len(report["matmul"]["sweeps"]) == len(embedded.matmul.shapes),
        "report sections do not cover the plan",
    )
    _check_patterns(report, embedded)
    _check_transposes(report, embedded)
    _check_matmul(report, embedded)
    expected = seal_report(build_analysis_report(embedded))
    _require(
        canonical_json_bytes(report) == canonical_json_bytes(expected),
        "analytical report does not match its rebuilt plan",
    )
    _check_lanes(report, embedded)
    return {
        "kind": ANALYSIS_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "plan_sha256": report["plan_sha256"],
        "patterns": len(report["patterns"]),
        "transposes": len(report["transposes"]),
        "tiled_estimates": report["summary"]["tiled_estimates"],
        "measured": False,
    }


def _benchmark(mapping: object) -> BenchmarkSpec:
    _require(
        type(mapping) is dict
        and set(mapping) == {"backends", "dtype", "shapes", "warmup", "repeats", "seed"}
        and type(mapping["backends"]) is list
        and type(mapping["shapes"]) is list,
        "benchmark fields do not match schema",
    )
    shapes = []
    for item in mapping["shapes"]:
        _require(
            type(item) is dict and set(item) == {"m", "n", "k"},
            "benchmark shape fields do not match schema",
        )
        shapes.append(MatmulShape(**item))
    spec = BenchmarkSpec(
        backends=tuple(mapping["backends"]),
        shapes=tuple(shapes),
        warmup=mapping["warmup"],
        repeats=mapping["repeats"],
        seed=mapping["seed"],
        dtype=mapping["dtype"],
    )
    _require(spec.to_mapping() == mapping, "benchmark section is not canonical")
    return spec


def _exact(actual: object, expected: object) -> bool:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 stay distinct."""

    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _finite(value: object, minimum: float = 0.0) -> bool:
    return type(value) is float and math.isfinite(value) and value >= minimum


def _nearest_rank(ordered: list[int], percentile: int) -> int:
    return ordered[(percentile * len(ordered) + 99) // 100 - 1]


def _tolerance(inner: int) -> float:
    single = inner * 2.0**-24
    double = inner * 2.0**-53
    return single / (1 - single) + double / (1 - double)


def _check_environment(environment: object) -> None:
    _require(
        type(environment) is dict and set(environment) == _ENVIRONMENT_FIELDS,
        "environment fields do not match the CPU record",
    )
    for name in _ENVIRONMENT_FIELDS - {"cpu_count", "torch_threads"}:
        value = environment[name]
        _require(
            isinstance(value, str) and _TOKEN.fullmatch(value) is not None,
            f"environment {name} must be a version or platform token",
        )
    count = environment["cpu_count"]
    _require(
        count is None or (type(count) is int and count >= 1),
        "environment cpu_count must be a positive integer or null",
    )
    threads = environment["torch_threads"]
    _require(type(threads) is int and threads >= 1, "torch_threads must be positive")


def _check_result(result: object, case, spec: BenchmarkSpec) -> None:
    _require(
        type(result) is dict and set(result) == _RESULT_FIELDS,
        "measurement result fields do not match schema",
    )
    shape = case.shape
    _require(
        _exact(
            [result["backend"], result["m"], result["n"], result["k"]],
            [case.backend, shape.m, shape.n, shape.k],
        )
        and result["device"] == "cpu"
        and result["dtype"] == spec.dtype
        and _exact(
            [result["warmup"], result["repeats"], result["seed"]],
            [spec.warmup, spec.repeats, spec.seed],
        ),
        "measurement result does not match its planned case",
    )
    _require(
        isinstance(result["input_sha256"], str)
        and _SHA256.fullmatch(result["input_sha256"]) is not None,
        "input_sha256 must be a SHA-256 hex digest",
    )
    samples = result["samples_ns"]
    _require(
        type(samples) is list
        and len(samples) == spec.repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        "samples_ns must hold one positive integer per repeat",
    )
    ordered = sorted(samples)
    timing = {
        "count": len(ordered),
        "min_ns": ordered[0],
        "p25_ns": _nearest_rank(ordered, 25),
        "median_ns": _nearest_rank(ordered, 50),
        "p75_ns": _nearest_rank(ordered, 75),
        "max_ns": ordered[-1],
    }
    timing["iqr_ns"] = timing["p75_ns"] - timing["p25_ns"]
    _require(_exact(result["timing"], timing), "timing statistics do not recompute")
    flops = 2 * shape.m * shape.n * shape.k
    _require(
        _exact(
            [
                result["flops"],
                result["median_gflops_per_second"],
                result["best_gflops_per_second"],
            ],
            [flops, flops / timing["median_ns"], flops / timing["min_ns"]],
        ),
        "FLOP and GFLOP/s accounting does not recompute",
    )
    correctness = result["correctness"]
    _require(
        type(correctness) is dict
        and set(correctness)
        == {
            "reference",
            "bound",
            "tolerance",
            "checked_results",
            "max_abs_error",
            "max_bound_ratio",
        }
        and correctness["reference"] == "float64"
        and correctness["bound"] == "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)"
        and _exact(correctness["tolerance"], _tolerance(shape.k))
        and _exact(correctness["checked_results"], spec.warmup + spec.repeats)
        and _finite(correctness["max_abs_error"])
        and _finite(correctness["max_bound_ratio"])
        and correctness["max_bound_ratio"] <= 1.0,
        "correctness evidence does not satisfy the float64 rounding bound",
    )


def _verify_measurement(
    report: object, plan: MemoryAccessPlan | None
) -> dict[str, object]:
    report = _seal_and_contract(report, _MEASUREMENT_FIELDS, MEASUREMENT_KIND)
    _require(report["measured"] is True, "measurement report must be measured")
    _require(
        report["measurement_scope"] == MEASUREMENT_SCOPE
        and report["evidence_boundary"] == MEASUREMENT_BOUNDARY,
        "measurement evidence boundary was weakened",
    )
    _require(
        report["timer"] == TIMER
        and report["percentile_method"] == PERCENTILE_METHOD
        and report["correctness_source"] == CORRECTNESS_SOURCE,
        "measurement method fields were changed",
    )
    _require(
        isinstance(report["plan_sha256"], str)
        and _SHA256.fullmatch(report["plan_sha256"]) is not None,
        "plan_sha256 must be a SHA-256 hex digest",
    )
    _require(
        isinstance(report["plan_name"], str)
        and _SLUG.fullmatch(report["plan_name"]) is not None,
        "plan_name must be a lowercase plan slug",
    )
    spec = _benchmark(report["benchmark"])
    if plan is not None:
        _require(
            report["plan_sha256"] == plan.fingerprint()
            and report["plan_name"] == plan.name
            and spec.to_mapping() == plan.benchmark.to_mapping(),
            "report was not measured from the supplied plan",
        )
    _check_environment(report["environment"])
    cases = spec.cases()
    results = report["results"]
    _require(
        type(results) is list and len(results) == len(cases),
        "measurement results do not cover the planned cases",
    )
    digests: dict[MatmulShape, str] = {}
    for result, case in zip(results, cases, strict=True):
        _check_result(result, case, spec)
        known = digests.setdefault(case.shape, result["input_sha256"])
        _require(
            known == result["input_sha256"],
            "backends measured different inputs for one shape",
        )
    _require(
        _exact(
            report["summary"],
            {
                "cases": len(results),
                "checked_results": sum(
                    item["correctness"]["checked_results"] for item in results
                ),
                "max_bound_ratio": max(
                    item["correctness"]["max_bound_ratio"] for item in results
                ),
            },
        ),
        "measurement summary does not reconcile",
    )
    return {
        "kind": MEASUREMENT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "plan_sha256": report["plan_sha256"],
        "cases": len(results),
        "measured": True,
        "scope": MEASUREMENT_SCOPE,
    }


def verify_report(
    report: object, *, plan: MemoryAccessPlan | None = None
) -> dict[str, object]:
    """Verify either report kind, optionally binding it to a supplied plan."""

    if plan is not None and not isinstance(plan, MemoryAccessPlan):
        raise TypeError("plan must be a MemoryAccessPlan")
    try:
        if type(report) is dict and report.get("kind") == MEASUREMENT_KIND:
            return _verify_measurement(report, plan)
        return _verify_analysis(report, plan)
    except MemoryAccessVerificationError:
        raise
    except (
        MemoryAccessError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
    ) as error:
        raise MemoryAccessVerificationError(
            f"report failed verification: {error}"
        ) from error
