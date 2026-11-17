"""Independent verification of sealed matmul model reports.

Verification trusts no number in a report. It checks the field set, the fixed
evidence boundary and the SHA-256 seal, parses the embedded plan, and can bind
the report to a plan file. It then recomputes the case, accounting, occupancy
and summary sections from the plan with this module's own arithmetic:

* inputs from the SHA-256 counter stream, held as exact integer significands;
* the stated matmul and every exact dot product, rounded to 24 bits with
  integer arithmetic and encoded as IEEE float32 bits without ``struct``;
* launch geometry by ceil division and the order digests, with the Day 73
  verifier's own permutation code instead of the simulator;
* every warp's lane addresses, sectors, lines and classes;
* the traffic models and roofline points as fractions;
* the occupancy from the register and shared memory rules, with the compute
  capability 8.0 values pinned here.

It reads the CUDA text with ``check_cuda_source`` and compares the limits,
the rule, the geometry and the architecture record with the lab constants.
The first field that differs is named. Last, the report must equal, byte for
byte, a fresh build that runs the simulator again.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_report import limits_mapping
from .kernel_launch_verify import (
    KernelLaunchVerificationError,
    _ceil,
    _compare,
    _geometry,
    _orders,
    _stream,
)
from .matmul_model_cuda import CHECK_NAMES, check_cuda_source
from .matmul_model_occupancy import architecture_mapping
from .matmul_model_plan import MatmulCase, MatmulModelPlan
from .matmul_model_reference import MatmulModelError
from .matmul_model_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    REPORT_KIND,
    SCHEMA_VERSION,
    build_report,
    geometry_mapping,
    report_digest,
    rule_mapping,
    seal_report,
)
from .memory_access_tiling import MatmulShape

MAX_REPORT_BYTES = 1024 * 1024
# Pinned here independently of the modules that the report uses.
_MAPPINGS = ("x_columns", "x_rows")
_STREAMS = ("matmul-a", "matmul-b")
_WARP = 32
_SECTOR = 32
_LINE = 128
_ELEMENT = 4
_PLACES = 6
_MAX_WARPS_PER_SM = 64
_MAX_BLOCKS_PER_SM = 32
_REGISTERS_PER_SM = 65_536
_REGISTER_UNIT = 256
_SUB_PARTITIONS = 4
_SHARED_PER_SM = 164 * 1024
_SHARED_UNIT = 128
_SHARED_RESERVED = 1024
_SHARED_PER_BLOCK = 48 * 1024
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "plan",
    "limits",
    "rule",
    "geometry",
    "cases",
    "accounting",
    "occupancy",
    "cuda_source",
    "summary",
    "evidence_boundary",
    DIGEST_FIELD,
}


class MatmulModelVerificationError(MatmulModelError):
    """Raised when a report does not reproduce from its declared plan."""


def _ratio(value: Fraction) -> dict[str, object]:
    scaled = round(value * 10**_PLACES)
    whole, part = divmod(abs(scaled), 10**_PLACES)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": f"{'-' if scaled < 0 else ''}{whole}.{part:0{_PLACES}d}",
    }


# A float32 value here is (significand, exponent): significand * 2**exponent,
# with |significand| below 2**24, or (0, 0) for +0.0.


def _operand(seed: int, stream: str, count: int) -> list[tuple[int, int]]:
    raw = _stream(seed, stream, 4 * count)
    values = []
    for index in range(count):
        word = int.from_bytes(raw[4 * index : 4 * index + 4], "big")
        significand = 0x800000 | (word & 0x7FFFFF)
        exponent = ((word >> 23) & 0xF) - 8 - 23
        values.append((-significand if word >> 31 else significand, exponent))
    return values


def _round24(value: int, exponent: int) -> tuple[int, int]:
    """Round ``value * 2**exponent`` to 24 significant bits, ties to even."""

    if value == 0:
        return 0, 0
    magnitude = abs(value)
    excess = magnitude.bit_length() - 24
    if excess > 0:
        quotient, remainder = divmod(magnitude, 1 << excess)
        half = 1 << (excess - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
        magnitude, exponent = quotient, exponent + excess
    return (-magnitude if value < 0 else magnitude), exponent


def _add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    low = min(left[1], right[1])
    total = (left[0] << (left[1] - low)) + (right[0] << (right[1] - low))
    return _round24(total, low)


def _bits(value: tuple[int, int]) -> int:
    """Encode a normal float32 value, or +0.0, as its IEEE 754 bit pattern."""

    significand, exponent = value
    if significand == 0:
        return 0
    magnitude = abs(significand)
    # A rounding carry leaves 2**24, whose low bit is 0, so the shift is exact.
    shift = magnitude.bit_length() - 24
    magnitude = magnitude >> shift if shift > 0 else magnitude << -shift
    biased = exponent + shift + 23 + 127
    if not 1 <= biased <= 254:
        raise MatmulModelVerificationError("a value left the normal float32 range")
    return (significand < 0) << 31 | biased << 23 | (magnitude & 0x7FFFFF)


def _digest(values: list[tuple[int, int]]) -> str:
    return sha256(
        b"".join(_bits(value).to_bytes(4, "little") for value in values)
    ).hexdigest()


def _ordered(bits: int) -> int:
    return -(bits & 0x7FFFFFFF) if bits >> 31 else bits


def _matmul(
    a: list[tuple[int, int]], b: list[tuple[int, int]], shape: MatmulShape
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the stated-order result and each exact dot rounded once."""

    m, n, k = shape.m, shape.n, shape.k
    stated, exact = [], []
    for row in range(m):
        for col in range(n):
            total = (0, 0)
            terms = []
            for i in range(k):
                left, right = a[row * k + i], b[i * n + col]
                terms.append((left[0] * right[0], left[1] + right[1]))
                total = _add(total, _round24(*terms[-1]))
            stated.append(total)
            low = min(exponent for _, exponent in terms)
            exact.append(
                _round24(
                    sum(value << (exponent - low) for value, exponent in terms), low
                )
            )
    return stated, exact


def _lane_elements(
    block: tuple[int, int], mapping: str, block_index: tuple[int, int], warp: int
) -> list[tuple[int, int]]:
    thread_y, first_x = divmod(warp * _WARP, block[0])
    y = block_index[1] * block[1] + thread_y
    lanes = []
    for lane in range(_WARP):
        x = block_index[0] * block[0] + first_x + lane
        lanes.append((y, x) if mapping == "x_columns" else (x, y))
    return lanes


def _request(addresses: list[int | None]) -> dict[str, object]:
    active = [
        (lane, address) for lane, address in enumerate(addresses) if address is not None
    ]
    distinct = {address for _, address in active}
    stride = None
    if len(active) > 1:
        (lane0, first), (lane1, second) = active[0], active[1]
        step, rest = divmod(second - first, lane1 - lane0)
        if not rest and all(
            address == first + (lane - lane0) * step for lane, address in active
        ):
            stride = step
    if len(distinct) == 1:
        kind = "broadcast"
    elif stride == _ELEMENT:
        kind = "coalesced"
    else:
        kind = "strided"
    return {
        "lane_addresses": list(addresses),
        "active_lanes": len(active),
        "distinct_addresses": len(distinct),
        "sectors": len({address // _SECTOR for address in distinct}),
        "lines": len({address // _LINE for address in distinct}),
        "class": kind,
        "lane_stride_bytes": stride,
    }


def _warp_requests(
    shape: MatmulShape, lanes: list[tuple[int, int]]
) -> tuple[list[dict], list[dict], dict] | None:
    m, n, k = shape.m, shape.n, shape.k
    active = [(row, col) if row < m and col < n else None for row, col in lanes]
    if all(lane is None for lane in active):
        return None

    def addresses(element) -> list[int | None]:
        return [None if lane is None else element(*lane) * _ELEMENT for lane in active]

    a_loads = [_request(addresses(lambda r, c, i=i: r * k + i)) for i in range(k)]
    b_loads = [_request(addresses(lambda r, c, i=i: i * n + c)) for i in range(k)]
    return a_loads, b_loads, _request(addresses(lambda r, c: r * n + c))


def _expected_access(
    plan: MatmulModelPlan, shape: MatmulShape, mapping: str, grid: list[int]
) -> tuple[dict[str, object], dict[str, object]]:
    totals = {
        name: {
            "requests": 0,
            "sectors": 0,
            "lines": 0,
            "requested_bytes": 0,
            "unique_bytes": 0,
            "transferred_bytes": 0,
            "classes": {"broadcast": 0, "coalesced": 0, "strided": 0},
        }
        for name in ("a", "b", "c")
    }
    warps = {"launched": 0, "full": 0, "partial": 0, "idle": 0}
    for block_y in range(grid[1]):
        for block_x in range(grid[0]):
            for warp in range(plan.block[0] * plan.block[1] // _WARP):
                warps["launched"] += 1
                lanes = _lane_elements(plan.block, mapping, (block_x, block_y), warp)
                requests = _warp_requests(shape, lanes)
                if requests is None:
                    warps["idle"] += 1
                    continue
                a_loads, b_loads, store = requests
                full = store["active_lanes"] == _WARP
                warps["full" if full else "partial"] += 1
                for name, items in (("a", a_loads), ("b", b_loads), ("c", [store])):
                    total = totals[name]
                    for item in items:
                        total["requests"] += 1
                        total["sectors"] += item["sectors"]
                        total["lines"] += item["lines"]
                        total["requested_bytes"] += item["active_lanes"] * _ELEMENT
                        total["unique_bytes"] += item["distinct_addresses"] * _ELEMENT
                        total["transferred_bytes"] += item["sectors"] * _SECTOR
                        total["classes"][item["class"]] += 1
    transferred = sum(total["transferred_bytes"] for total in totals.values())
    access = {
        "warps": warps,
        "operands": totals,
        "transferred_bytes": transferred,
        "flops_per_transferred_byte": _ratio(Fraction(shape.flops, transferred)),
    }
    classes = {name: warps[name] for name in ("full", "partial", "idle")}
    requests = sum(total["requests"] for total in totals.values())
    cross_check = {
        "memory_access_requests": requests,
        "memory_access_agreeing": requests,
        "mismatched_warps": 0,
        "warp_classes": classes,
        "simt_warp_classes": dict(classes),
        "passed": True,
    }
    return access, cross_check


def _expected_probes(
    plan: MatmulModelPlan, shape: MatmulShape, mapping: str
) -> list[dict[str, object]]:
    probes = []
    seen = set()
    for row, col in ((0, 0), (shape.m - 1, shape.n - 1)):
        x, y = (col, row) if mapping == "x_columns" else (row, col)
        block_index = (x // plan.block[0], y // plan.block[1])
        warp = (x % plan.block[0] + plan.block[0] * (y % plan.block[1])) // _WARP
        if (block_index, warp) in seen:
            continue
        seen.add((block_index, warp))
        lanes = _lane_elements(plan.block, mapping, block_index, warp)
        a_loads, b_loads, store = _warp_requests(shape, lanes)
        probes.append(
            {
                "element": [row, col],
                "warp": {"block": [*block_index, 0], "warp": warp},
                "active_lanes": store["active_lanes"],
                "steps": [
                    {"step": i, "a": a_load, "b": b_load}
                    for i, (a_load, b_load) in enumerate(zip(a_loads, b_loads))
                ],
                "c_store": store,
                "trace_matches_model": True,
            }
        )
    return probes


def _expected_mapping(
    plan: MatmulModelPlan, case: MatmulCase, mapping: str, reference: str
) -> dict[str, object]:
    shape = case.shape
    elements = shape.m * shape.n
    extent = [shape.n, shape.m] if mapping == "x_columns" else [shape.m, shape.n]
    grid = [_ceil(extent[0], plan.block[0]), _ceil(extent[1], plan.block[1]), 1]
    geometry = _geometry(grid, [*plan.block, 1])
    launched = geometry["launched_threads"]
    access, cross_check = _expected_access(plan, shape, mapping, grid)
    return {
        "mapping": mapping,
        "kernel": f"naive_matmul_{mapping}",
        "extent": [*extent, 1],
        "ragged": extent[0] % plan.block[0] != 0 or extent[1] % plan.block[1] != 0,
        "launch": {
            **geometry,
            "useful_threads": elements,
            "idle_threads": launched - elements,
        },
        "runs": [
            {
                "order": order,
                "order_sha256": digest,
                "threads_run": launched,
                "useful_threads": elements,
                "output_sha256": reference,
                "matches_reference": True,
                "reads": 2 * elements * shape.k,
                "writes": elements,
                "out_of_bounds_reads": 0,
                "out_of_bounds_writes": 0,
                "write_conflicts": 0,
                "read_write_conflicts": 0,
            }
            for order, digest in _orders(plan, launched)
        ],
        "order_independent": True,
        "access": access,
        "cross_check": cross_check,
        "probes": _expected_probes(plan, shape, mapping),
    }


def _expected_case(plan: MatmulModelPlan, case: MatmulCase) -> dict[str, object]:
    shape = case.shape
    a = _operand(case.input_seed, _STREAMS[0], shape.m * shape.k)
    b = _operand(case.input_seed, _STREAMS[1], shape.k * shape.n)
    stated, exact = _matmul(a, b, shape)
    distances = [
        abs(_ordered(_bits(left)) - _ordered(_bits(right)))
        for left, right in zip(stated, exact)
    ]
    reference = _digest(stated)
    return {
        "name": case.name,
        "shape": {"m": shape.m, "n": shape.n, "k": shape.k},
        "inputs_sha256": {"a": _digest(a), "b": _digest(b)},
        "reference_sha256": reference,
        "order_rounding": {
            "elements": len(stated),
            "differing": sum(distance > 0 for distance in distances),
            "max_ulp_difference": max(distances),
            "correctly_rounded_sha256": _digest(exact),
        },
        "mappings": [
            _expected_mapping(plan, case, mapping, reference) for mapping in _MAPPINGS
        ],
    }


def _point(intensity: Fraction, ceilings: Mapping[str, str]) -> dict[str, object]:
    compute = Fraction(ceilings["compute_gflops"])
    bandwidth = Fraction(ceilings["bandwidth_gbps"])
    memory = intensity * bandwidth
    attainable = min(compute, memory)
    bound = "memory" if memory < compute else "compute" if memory > compute else "ridge"
    return {
        "intensity": _ratio(intensity),
        "ridge_point": _ratio(compute / bandwidth),
        "attainable_gflops": _ratio(attainable),
        "bound": bound,
        "fraction_of_compute_ceiling": _ratio(attainable / compute),
    }


def _expected_accounting(
    plan: MatmulModelPlan, shape: MatmulShape, source: str
) -> dict[str, object]:
    m, n, k = shape.m, shape.n, shape.k
    flops = 2 * m * n * k
    models = {
        "no_reuse": (m * n * k, m * n * k, m * n),
        "ideal_reuse": (m * k, k * n, m * n),
    }
    sections = {}
    intensities = {}
    for name, (a_count, b_count, c_count) in models.items():
        total = (a_count + b_count + c_count) * _ELEMENT
        intensities[name] = Fraction(flops, total)
        sections[name] = {
            "a_elements": a_count,
            "b_elements": b_count,
            "c_elements": c_count,
            "bytes": total,
            "intensity": _ratio(intensities[name]),
        }
    ceilings = [item.to_mapping() for item in plan.ceilings]
    return {
        "source": source,
        "shape": {"m": m, "n": n, "k": k},
        "flops": flops,
        **sections,
        "no_reuse_intensity_limit": _ratio(Fraction(1, 4)),
        "reuse_factor": _ratio(
            Fraction(sections["no_reuse"]["bytes"], sections["ideal_reuse"]["bytes"])
        ),
        "cross_checks": {
            "memory_access_naive_traffic": True,
            "roofline_matmul_workload": True,
        },
        "roofline": [
            {
                "ceilings": item,
                "no_reuse": _point(intensities["no_reuse"], item),
                "ideal_reuse": _point(intensities["ideal_reuse"], item),
            }
            for item in ceilings
        ],
    }


def _expected_estimate(threads: int, registers: int) -> dict[str, object]:
    warps = _ceil(threads, _WARP)
    per_warp = _ceil(registers * _WARP, _REGISTER_UNIT) * _REGISTER_UNIT
    per_sub_partition = _REGISTERS_PER_SM // _SUB_PARTITIONS // per_warp
    register_blocks = per_sub_partition * _SUB_PARTITIONS // warps
    allocated = _ceil(_SHARED_RESERVED, _SHARED_UNIT) * _SHARED_UNIT
    limits = {
        "warps": _MAX_WARPS_PER_SM // warps,
        "registers": register_blocks,
        "shared_memory": _SHARED_PER_SM // allocated,
        "blocks": _MAX_BLOCKS_PER_SM,
    }
    active = min(limits.values())
    return {
        "compute_capability": "8.0",
        "kernel": {
            "threads_per_block": threads,
            "registers_per_thread": registers,
            "static_shared_bytes": 0,
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
            "requested_bytes_per_block": _SHARED_RESERVED,
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


def _expected_cuda(plan: MatmulModelPlan, section: object) -> dict[str, object]:
    if not isinstance(section, Mapping) or type(section.get("text")) is not str:
        raise MatmulModelVerificationError("cuda_source must hold the source text")
    text = section["text"]
    if not text.isascii():
        raise MatmulModelVerificationError("cuda_source text must be ASCII")
    structures = []
    for case in plan.cases:
        shape = case.shape
        expected = {
            "checks": {name: True for name in CHECK_NAMES},
            "passed": True,
            "launch": {
                "grid": [
                    _ceil(shape.n, plan.block[0]),
                    _ceil(shape.m, plan.block[1]),
                    1,
                ],
                "block": [*plan.block, 1],
            },
        }
        check = check_cuda_source(text, shape=shape, block=plan.block)
        _compare(
            expected, check.to_mapping(), f"cuda_source.text structure for {case.name}"
        )
        structures.append({"case": case.name, **expected})
    return {
        "language": "CUDA C++",
        "compiled": False,
        "executed": False,
        "sha256": sha256(text.encode("ascii")).hexdigest(),
        "text": text,
        "structure_by_case": structures,
    }


def _expected_summary(sections: Mapping[str, object]) -> dict[str, object]:
    launches = [item for case in sections["cases"] for item in case["mappings"]]
    runs = [run for item in launches for run in item["runs"]]
    probes = [probe for item in launches for probe in item["probes"]]
    points = [
        entry[model]
        for item in sections["accounting"]
        for entry in item["roofline"]
        for model in ("no_reuse", "ideal_reuse")
    ]
    structures = sections["cuda_source"]["structure_by_case"]
    return {
        "cases": len(sections["cases"]),
        "launches": len(launches),
        "ragged_launches": sum(item["ragged"] for item in launches),
        "runs": len(runs),
        "runs_matching_reference": len(runs),
        "memory_safe_runs": len(runs),
        "race_free_runs": len(runs),
        "order_independent_launches": len(launches),
        "cross_checked_launches": len(launches),
        "cross_checked_requests": sum(
            item["cross_check"]["memory_access_requests"] for item in launches
        ),
        "probes": len(probes),
        "probe_traces_matching_model": len(probes),
        "accounting_shapes": len(sections["accounting"]),
        "roofline_points": len(points),
        "memory_bound_points": sum(point["bound"] == "memory" for point in points),
        "occupancy_estimates": len(sections["occupancy"]["estimates"]),
        "cuda_checks_passed": len(CHECK_NAMES) * len(structures),
        "cuda_checks": len(CHECK_NAMES) * len(structures),
    }


def _verify(
    report: Mapping[str, object], plan: MatmulModelPlan | None
) -> MatmulModelPlan:
    if type(report["schema_version"]) is not int or (
        report["schema_version"] != SCHEMA_VERSION
    ):
        raise MatmulModelVerificationError("unsupported report schema_version")
    if report["kind"] != REPORT_KIND:
        raise MatmulModelVerificationError(f"report kind is not {REPORT_KIND}")
    if report[DIGEST_FIELD] != report_digest(report):
        raise MatmulModelVerificationError("report digest does not match its content")
    _compare(dict(EVIDENCE_BOUNDARY), report["evidence_boundary"], "evidence_boundary")
    try:
        embedded = MatmulModelPlan.from_mapping(report["plan"])
    except (ValueError, TypeError) as error:
        raise MatmulModelVerificationError(
            f"embedded plan is invalid: {error}"
        ) from error
    _compare(embedded.to_mapping(), report["plan"], "plan")
    if plan is not None:
        if type(plan) is not MatmulModelPlan:
            raise TypeError("plan must be MatmulModelPlan")
        if plan != embedded:
            raise MatmulModelVerificationError("report was not built from this plan")
    expected: dict[str, object] = {
        "limits": limits_mapping(),
        "rule": rule_mapping(),
        "geometry": geometry_mapping(),
        "cases": [_expected_case(embedded, case) for case in embedded.cases],
        "accounting": [
            *(
                _expected_accounting(embedded, case.shape, f"case:{case.name}")
                for case in embedded.cases
            ),
            *(
                _expected_accounting(embedded, shape, "declared")
                for shape in embedded.accounting_shapes
            ),
        ],
        "occupancy": {
            "architecture": architecture_mapping(),
            "threads_per_block": embedded.threads_per_block,
            "estimates": [
                _expected_estimate(embedded.threads_per_block, count)
                for count in embedded.registers_per_thread
            ],
        },
        "cuda_source": _expected_cuda(embedded, report["cuda_source"]),
    }
    expected["summary"] = _expected_summary(expected)
    for name, section in expected.items():
        _compare(section, report[name], name)
    rebuilt = seal_report(build_report(embedded))
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(report)):
        raise MatmulModelVerificationError("report differs from a fresh build")
    return embedded


def verify_report(
    report: object, *, plan: MatmulModelPlan | None = None
) -> MatmulModelPlan:
    """Return the report's plan only if every field reproduces exactly."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise MatmulModelVerificationError("report fields do not match the schema")
    try:
        return _verify(report, plan)
    except KernelLaunchVerificationError as error:
        # The Day 73 comparison helper names the field; keep its message.
        raise MatmulModelVerificationError(str(error)) from error


def load_report(
    path: Path, *, plan: MatmulModelPlan | None = None
) -> dict[str, object]:
    """Load canonical report bytes and verify every field.

    A file that cannot be read raises ``KernelLaunchError``. A file that was
    read but is not a canonical, reproducible report raises
    ``MatmulModelVerificationError``.
    """

    payload = read_bounded(path, limit=MAX_REPORT_BYTES, label="report")
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise MatmulModelVerificationError(str(error)) from error
    if type(report) is not dict:
        raise MatmulModelVerificationError("report must be a JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise MatmulModelVerificationError("report bytes are not canonical JSON")
    verify_report(report, plan=plan)
    return report
