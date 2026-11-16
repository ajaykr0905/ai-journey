"""Independent verification of sealed kernel-launch reports.

Verification trusts no number in a report. It checks the field set, the
fixed evidence boundary and the SHA-256 seal, parses the embedded plan, and
can bind the report to a plan file. It then recomputes every section from
the plan with this module's own code: launch geometry by ceil division, input
streams from SHA-256 counter blocks, float32 sums through ``array('f')``, the
grayscale rule in integers, permutation digests, and the out-of-bounds
records and write conflicts of each unguarded demo by direct enumeration of
its threads, without the simulator. The first field that differs is named.
Last, the report must equal, byte for byte, a fresh build that runs the
simulator again.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_cuda import CHECK_NAMES, CUDA_LANGUAGE, check_cuda_source
from .kernel_launch_io import (
    canonical_json_bytes,
    canonical_sha256,
    loads_strict,
    read_bounded,
)
from .kernel_launch_plan import KernelLaunchPlan
from .kernel_launch_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    FIRST_RECORDS,
    REPORT_KIND,
    SCHEMA_VERSION,
    build_report,
    grayscale_rule_mapping,
    limits_mapping,
    report_digest,
    seal_report,
)

MAX_REPORT_BYTES = 1024 * 1024
# The stream labels are pinned here independently of the modules that use them.
_INPUT_LABEL = "ai-journey-kernel-launch-input"
_ORDER_LABEL = "ai-journey-kernel-launch-order"
_WEIGHTS_PER_10000 = (2989, 5870, 1140)
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "plan",
    "limits",
    "grayscale_rule",
    "vector_add",
    "rgb_to_grayscale",
    "unguarded_demos",
    "cuda_source",
    "summary",
    "evidence_boundary",
    DIGEST_FIELD,
}


class KernelLaunchVerificationError(KernelLaunchError):
    """Raised when a report does not reproduce from its declared plan."""


def _compare(expected: object, actual: object, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise KernelLaunchVerificationError(f"{path} fields do not match")
        for key in sorted(expected):
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise KernelLaunchVerificationError(f"{path} rows do not match")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise KernelLaunchVerificationError(
            f"{path} does not match independent recomputation"
        )


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _stream(seed: int, stream: str, count: int) -> bytes:
    data = bytearray()
    counter = 0
    while len(data) < count:
        data += sha256(f"{_INPUT_LABEL}:{stream}:{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(data[:count])


def _float32_stream(seed: int, stream: str, count: int) -> list[float]:
    raw = _stream(seed, stream, 4 * count)
    values = []
    for index in range(count):
        word = int.from_bytes(raw[4 * index : 4 * index + 4], "big")
        sign = -1.0 if word >> 31 else 1.0
        exponent = ((word >> 23) & 0xF) - 8
        values.append(sign * math.ldexp(1.0 + (word & 0x7FFFFF) / 2**23, exponent))
    return values


def _float32_digest(values: list[float]) -> str:
    packed = array("f", values)
    if sys.byteorder == "big":
        packed.byteswap()
    return sha256(packed.tobytes()).hexdigest()


def _permutation(seed: int, count: int) -> list[int]:
    order = list(range(count))
    for index in range(count - 1, 0, -1):
        bound = index + 1
        limit = 2**64 - 2**64 % bound
        attempt = 0
        while True:
            digest = sha256(f"{_ORDER_LABEL}:{seed}:{index}:{attempt}".encode())
            value = int.from_bytes(digest.digest()[:8], "big")
            if value < limit:
                break
            attempt += 1
        other = value % bound
        order[index], order[other] = order[other], order[index]
    return order


def _order_digest(order: list[int]) -> str:
    return sha256(b"".join(item.to_bytes(8, "big") for item in order)).hexdigest()


def _orders(plan: KernelLaunchPlan, launched: int) -> list[tuple[dict, str]]:
    orders = [
        ({"kind": "canonical", "seed": None}, _order_digest(list(range(launched))))
    ]
    for seed in plan.permuted_order_seeds:
        orders.append(
            (
                {"kind": "permuted", "seed": seed},
                _order_digest(_permutation(seed, launched)),
            )
        )
    return orders


def _geometry(grid: list[int], block: list[int]) -> dict[str, object]:
    threads = block[0] * block[1] * block[2]
    blocks = grid[0] * grid[1] * grid[2]
    warps = _ceil(threads, 32)
    return {
        "grid": grid,
        "block": block,
        "threads_per_block": threads,
        "blocks": blocks,
        "launched_threads": threads * blocks,
        "warps_per_block": warps,
        "idle_lanes_per_block": warps * 32 - threads,
    }


def _luma(red: int, green: int, blue: int) -> int:
    total = sum(w * c for w, c in zip(_WEIGHTS_PER_10000, (red, green, blue)))
    return min(255, (total + 5000) // 10000)


def _gray(rgb: bytes) -> list[int]:
    return [_luma(*rgb[start : start + 3]) for start in range(0, len(rgb), 3)]


def _runs(
    plan: KernelLaunchPlan,
    launched: int,
    useful: int,
    reads: int,
    output: str,
) -> list[dict[str, object]]:
    return [
        {
            "order": order,
            "order_sha256": digest,
            "threads_run": launched,
            "useful_threads": useful,
            "output_sha256": output,
            "matches_reference": True,
            "reads": reads,
            "writes": useful,
            "out_of_bounds_reads": 0,
            "out_of_bounds_writes": 0,
            "write_conflicts": 0,
            "read_write_conflicts": 0,
        }
        for order, digest in _orders(plan, launched)
    ]


def _vector_expected(
    plan: KernelLaunchPlan, n: int
) -> tuple[list[float], list[float], list[float]]:
    a = _float32_stream(plan.vector_add.input_seed, "vector-add-a", n)
    b = _float32_stream(plan.vector_add.input_seed, "vector-add-b", n)
    return a, b, list(array("f", [x + y for x, y in zip(a, b)]))


def _expected_vector_add(plan: KernelLaunchPlan) -> dict[str, object]:
    case = plan.vector_add
    a, b, sums = _vector_expected(plan, case.n)
    reference = _float32_digest(sums)
    grid = [_ceil(case.n, case.threads_per_block), 1, 1]
    geometry = _geometry(grid, [case.threads_per_block, 1, 1])
    launched = geometry["launched_threads"]
    return {
        "inputs_sha256": {"a": _float32_digest(a), "b": _float32_digest(b)},
        "rounded_sums": sum(
            Fraction(total) != Fraction(x) + Fraction(y)
            for total, x, y in zip(sums, a, b)
        ),
        "extent": [case.n, 1, 1],
        "launch": {
            **geometry,
            "useful_threads": case.n,
            "idle_threads": launched - case.n,
        },
        "reference_sha256": reference,
        "runs": _runs(plan, launched, case.n, 2 * case.n, reference),
        "order_independent": True,
    }


def _expected_grayscale(plan: KernelLaunchPlan) -> dict[str, object]:
    case = plan.rgb_to_grayscale
    pixels = case.width * case.height
    rgb = _stream(case.input_seed, "rgb-image", 3 * pixels)
    reference = sha256(bytes(_gray(rgb))).hexdigest()
    totals = [
        sum(w * c for w, c in zip(_WEIGHTS_PER_10000, rgb[start : start + 3]))
        for start in range(0, len(rgb), 3)
    ]
    block_x, block_y = case.block
    grid = [_ceil(case.width, block_x), _ceil(case.height, block_y), 1]
    geometry = _geometry(grid, [block_x, block_y, 1])
    launched = geometry["launched_threads"]
    return {
        "inputs_sha256": {"rgb": sha256(rgb).hexdigest()},
        "rounding": {
            "ties": sum(total % 10000 == 5000 for total in totals),
            "truncation_differs": sum(total % 10000 >= 5000 for total in totals),
        },
        "extent": [case.width, case.height, 1],
        "launch": {
            **geometry,
            "useful_threads": pixels,
            "idle_threads": launched - pixels,
        },
        "reference_sha256": reference,
        "runs": _runs(plan, launched, pixels, 3 * pixels, reference),
        "order_independent": True,
    }


def _record(buffer: str, access: str, index: int, block: list, thread: list) -> dict:
    return {
        "buffer": buffer,
        "access": access,
        "index": index,
        "block": block,
        "thread": thread,
    }


def _expected_demo(plan: KernelLaunchPlan, kernel: str, extent: tuple) -> dict:
    """Enumerate an unguarded launch thread by thread, without the simulator."""

    records: list[dict] = []
    writers: dict[int, list[dict]] = {}
    if kernel == "vector_add":
        (n,) = extent
        width, height = n, 1
        block = [plan.vector_add.threads_per_block, 1, 1]
        reference = _float32_digest(_vector_expected(plan, n)[2])
        loads = (("a", 1, 0), ("b", 1, 0))
        store = "out"
    else:
        width, height = extent
        block = [*plan.rgb_to_grayscale.block, 1]
        rgb = _stream(plan.rgb_to_grayscale.input_seed, "rgb-image", 3 * width * height)
        reference = sha256(bytes(_gray(rgb))).hexdigest()
        loads = (("rgb", 3, 0), ("rgb", 3, 1), ("rgb", 3, 2))
        store = "gray"
    grid = [_ceil(width, block[0]), _ceil(height, block[1]), 1]
    elements = width * height
    for block_y in range(grid[1]):
        for block_x in range(grid[0]):
            for thread_y in range(block[1]):
                for thread_x in range(block[0]):
                    where = {
                        "block": [block_x, block_y, 0],
                        "thread": [thread_x, thread_y, 0],
                    }
                    column = block_x * block[0] + thread_x
                    row = block_y * block[1] + thread_y
                    index = row * width + column
                    for buffer, stride, offset in loads:
                        if stride * index + offset >= stride * elements:
                            records.append(
                                _record(
                                    buffer, "read", stride * index + offset, **where
                                )
                            )
                    if index >= elements:
                        records.append(_record(store, "write", index, **where))
                    else:
                        writers.setdefault(index, []).append(where)
    conflicts = [
        {"buffer": store, "index": index, "writers": writers[index]}
        for index in sorted(writers)
        if len(writers[index]) > 1
    ]
    return {
        "kernel": f"{kernel}_unguarded",
        "extent": [width, height, 1],
        "ragged": width % block[0] != 0 or height % block[1] != 0,
        "launch": _geometry(grid, block),
        "orders_checked": 1 + len(plan.permuted_order_seeds),
        "out_of_bounds_reads": sum(item["access"] == "read" for item in records),
        "out_of_bounds_writes": sum(item["access"] == "write" for item in records),
        "write_conflicts": len(conflicts),
        "extra_writes": sum(len(item["writers"]) - 1 for item in conflicts),
        "read_write_conflicts": 0,
        "out_of_bounds_sha256": canonical_sha256(records),
        "conflicts_sha256": canonical_sha256({"write": conflicts, "read_write": []}),
        "first_out_of_bounds_writes": [
            item for item in records if item["access"] == "write"
        ][:FIRST_RECORDS],
        "output_sha256": reference,
        "in_bounds_output_matches_reference": True,
        "orders_agree": True,
    }


def _expected_cuda(plan: KernelLaunchPlan, section: object) -> dict[str, object]:
    if not isinstance(section, Mapping) or type(section.get("text")) is not str:
        raise KernelLaunchVerificationError("cuda_source must hold the source text")
    text = section["text"]
    if not text.isascii():
        raise KernelLaunchVerificationError("cuda_source text must be ASCII")
    vector, gray = plan.vector_add, plan.rgb_to_grayscale
    check = check_cuda_source(
        text,
        n=vector.n,
        threads_per_block=vector.threads_per_block,
        width=gray.width,
        height=gray.height,
        block=gray.block,
    )
    expected_launches = {
        "vector_add": {
            "grid": [_ceil(vector.n, vector.threads_per_block), 1, 1],
            "block": [vector.threads_per_block, 1, 1],
        },
        "rgb_to_grayscale": {
            "grid": [
                _ceil(gray.width, gray.block[0]),
                _ceil(gray.height, gray.block[1]),
                1,
            ],
            "block": [*gray.block, 1],
        },
    }
    structure = {
        "checks": {name: True for name in CHECK_NAMES},
        "passed": True,
        "launches": expected_launches,
    }
    _compare(structure, check.to_mapping(), "cuda_source.text structure")
    return {
        "language": CUDA_LANGUAGE,
        "compiled": False,
        "executed": False,
        "sha256": sha256(text.encode("ascii")).hexdigest(),
        "text": text,
        "structure": structure,
    }


def _expected_summary(sections: dict[str, object]) -> dict[str, object]:
    kernels = (sections["vector_add"], sections["rgb_to_grayscale"])
    demos = sections["unguarded_demos"]
    runs = [run for section in kernels for run in section["runs"]]
    ragged = [demo for demo in demos if demo["ragged"]]
    aligned = [demo for demo in demos if not demo["ragged"]]
    checks = sections["cuda_source"]["structure"]["checks"]
    return {
        "kernels": 2,
        "runs": len(runs),
        "runs_matching_reference": len(runs),
        "race_free_runs": len(runs),
        "order_independent_kernels": 2,
        "unguarded_demos": len(demos),
        "ragged_demos_with_out_of_bounds_writes": sum(
            demo["out_of_bounds_writes"] > 0 for demo in ragged
        ),
        "ragged_demos": len(ragged),
        "aligned_demos_without_misuse": sum(
            demo["out_of_bounds_reads"] + demo["out_of_bounds_writes"] == 0
            and demo["write_conflicts"] == 0
            for demo in aligned
        ),
        "aligned_demos": len(aligned),
        "out_of_bounds_writes_caught": sum(
            demo["out_of_bounds_writes"] for demo in demos
        ),
        "write_conflicts_caught": sum(demo["write_conflicts"] for demo in demos),
        "cuda_checks_passed": len(checks),
        "cuda_checks": len(checks),
    }


def verify_report(
    report: object, *, plan: KernelLaunchPlan | None = None
) -> KernelLaunchPlan:
    """Return the report's plan only if every field reproduces exactly."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise KernelLaunchVerificationError("report fields do not match the schema")
    if type(report["schema_version"]) is not int or (
        report["schema_version"] != SCHEMA_VERSION
    ):
        raise KernelLaunchVerificationError("unsupported report schema_version")
    if report["kind"] != REPORT_KIND:
        raise KernelLaunchVerificationError(f"report kind is not {REPORT_KIND}")
    if report[DIGEST_FIELD] != report_digest(report):
        raise KernelLaunchVerificationError("report digest does not match its content")
    _compare(dict(EVIDENCE_BOUNDARY), report["evidence_boundary"], "evidence_boundary")
    try:
        embedded = KernelLaunchPlan.from_mapping(report["plan"])
    except (KernelLaunchError, TypeError) as error:
        raise KernelLaunchVerificationError(
            f"embedded plan is invalid: {error}"
        ) from error
    _compare(embedded.to_mapping(), report["plan"], "plan")
    if plan is not None:
        if type(plan) is not KernelLaunchPlan:
            raise TypeError("plan must be KernelLaunchPlan")
        if plan != embedded:
            raise KernelLaunchVerificationError("report was not built from this plan")
    expected: dict[str, object] = {
        "limits": limits_mapping(),
        "grayscale_rule": grayscale_rule_mapping(),
        "vector_add": _expected_vector_add(embedded),
        "rgb_to_grayscale": _expected_grayscale(embedded),
        "unguarded_demos": [
            _expected_demo(embedded, demo.kernel, demo.extent)
            for demo in embedded.unguarded_demos
        ],
        "cuda_source": _expected_cuda(embedded, report["cuda_source"]),
    }
    expected["summary"] = _expected_summary(expected)
    for name, section in expected.items():
        _compare(section, report[name], name)
    rebuilt = seal_report(build_report(embedded))
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(report)):
        raise KernelLaunchVerificationError("report differs from a fresh build")
    return embedded


def load_report(
    path: Path, *, plan: KernelLaunchPlan | None = None
) -> dict[str, object]:
    """Load canonical report bytes and verify every field.

    A file that cannot be read raises ``KernelLaunchError``. A file that was
    read but is not a canonical, reproducible report raises
    ``KernelLaunchVerificationError``.
    """

    payload = read_bounded(path, limit=MAX_REPORT_BYTES, label="report")
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise KernelLaunchVerificationError(str(error)) from error
    if type(report) is not dict:
        raise KernelLaunchVerificationError("report must be a JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise KernelLaunchVerificationError("report bytes are not canonical JSON")
    verify_report(report, plan=plan)
    return report
