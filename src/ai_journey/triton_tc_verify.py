"""Independent verification of sealed Day 82 reports, on the CPU.

The verifier checks the field set, the label and its evidence boundary and
the SHA-256 seal. Then it rebuilds every section with other code than the
lab's:

* the kernel section field by field from a table written here, and the
  kernel digest with ``hashlib`` over ``TRITON_SOURCE`` against a digest
  pinned here;
* the config from the raw file bytes with plain ``json``, and its digest;
* the whole model from the raw config mapping, in exact fractions, with its
  own formulas and its own half-to-even rounding;
* the verdict of each compile row from the counts that the row records,
  with a rule table and an opcode grammar written here;
* the field set of every record in a row, each MMA form from its opcode,
  and the counts that the lab derives from other counts of the row;
* with a dump folder, the index against the report, every text by its
  digest, line count and size, and the whole PTX and TTGIR record of each
  row from its text, with expressions and a parameter parser written here.

Without a dump folder, the verifier cannot read the texts, so it checks only
that each row is complete and agrees with its own counts. The verifier
imports neither triton nor torch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_verify import KernelLaunchVerificationError, _compare
from .triton_tc_kernel import TRITON_SOURCE, TritonTcError
from .triton_tc_model import LABEL as MODEL_TEXT_LABEL
from .triton_tc_model import OMITTED, RULE
from .triton_tc_report import (
    BOUNDARIES,
    COMPILER_LABEL,
    DIGEST_FIELD,
    LABELS,
    MODEL_LABEL,
    REPORT_KIND,
    SCHEMA_VERSION,
)

MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
# Pinned here independently of the modules that build the report.
_KERNEL_SHA256 = "6833430d2fea2ac4557766b4fef78b335d34d97b513058c6ec1ad7c2ee84fd9a"
_PATHS = ("fp32", "tf32", "fp16", "bf16")
_INPUT = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2}
_VARIANTS = ("fp32-ieee", "fp32-tf32", "fp16", "bf16")
_TARGETS = ("sm_80", "sm_90")
_ORDER = [(variant, target) for variant in _VARIANTS for target in _TARGETS]
_OPERAND = {"fp32-ieee": None, "fp32-tf32": "tf32", "fp16": "f16", "bf16": "bf16"}
_TILE = 128 * 128 * 32
_THREADS = 128
_WARPS = 4
_GROUPS = 1
_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i64": 64, "i32": 32, "i8": 8, "i1": 1}
_MMA = re.compile(
    r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.(row|col)\.(row|col)"
    r"(?:\.satfinite)?\.(\w+)\.(\w+)\.(\w+)\.(\w+)"
)
_WGMMA = re.compile(
    r"wgmma\.mma_async(?:\.sp)?\.sync\.aligned\.m(\d+)n(\d+)k(\d+)"
    r"(?:\.satfinite)?\.(\w+)\.(\w+)\.(\w+)"
)
# An opcode may hold "::", as in shared::cta, but it never ends in ":", so a
# label such as "loop:" is not an opcode.
_STATEMENT = re.compile(
    r"(?:^|[;{}\n])[ \t]*(?:[A-Za-z_$][\w$]*:[ \t]*)*(?:@!?%?\w+[ \t]+)?"
    r"([a-z](?:[\w.]|::)*)(?![\w.:])"
)
# PTX comments follow C. The token that starts first wins: a string literal,
# a line comment, or a block comment that can span lines.
_PTX_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|//[^\n]*|/\*.*?\*/', re.S)
_CLASS_PREFIXES = (
    ("mma.sync.aligned.", "mma_sync"),
    ("wgmma.mma_async.", "wgmma_mma_async"),
    ("ldmatrix.sync.aligned.", "ldmatrix"),
    ("stmatrix.sync.aligned.", "stmatrix"),
    ("cp.async.bulk.", "cp_async_bulk"),
    ("cp.async.ca.", "cp_async"),
    ("cp.async.cg.", "cp_async"),
)
_CLASS_EXACT = {
    "wgmma.fence.sync.aligned": "wgmma_fence",
    "wgmma.commit_group.sync.aligned": "wgmma_commit_group",
    "wgmma.wait_group.sync.aligned": "wgmma_wait_group",
    "cp.async.commit_group": "cp_async_commit_group",
    "cp.async.wait_group": "cp_async_wait_group",
    "cp.async.wait_all": "cp_async_wait_group",
    "fma.rn.f32": "fma_rn_f32",
}
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "label",
    "evidence_boundary",
    "kernel",
    "config",
    "config_sha256",
    "model",
    "compile",
    "recorded_at",
    "host_load",
    "summary",
    DIGEST_FIELD,
}
_TTGIR_OPS = (
    "tt.dot",
    "ttng.warp_group_dot",
    "ttng.warp_group_dot_wait",
    "ttg.local_alloc",
    "ttg.local_load",
    "ttg.local_store",
    "ttg.async_copy_global_to_local",
    "ttg.async_commit_group",
    "ttg.async_wait",
    "ttg.convert_layout",
    "ttng.fence_async_shared",
    "ttng.async_tma_copy_global_to_local",
    "scf.for",
)
_CLASSES = (
    "mma_sync",
    "wgmma_mma_async",
    "wgmma_fence",
    "wgmma_commit_group",
    "wgmma_wait_group",
    "ldmatrix",
    "stmatrix",
    "cp_async",
    "cp_async_commit_group",
    "cp_async_wait_group",
    "cp_async_bulk",
    "cvt_tf32",
    "fma_rn_f32",
)
_FAMILIES = ("mma_sync", "wgmma_mma_async")
_MMA_KINDS = ("nvidia_mma", "mma")
_SHARED_KINDS = ("swizzled_shared", "nvmma_shared", "shared", "shared_linear")
_DOT_OPS = ("tt.dot", "ttng.warp_group_dot")
# The record of each compile row, field by field, as the lab writes it.
_ROW_FIELDS = {
    "variant",
    "target",
    "expected",
    "allowed",
    "observed",
    "checks",
    "accepted",
    "note",
    "shared_bytes",
    "largest_buffer_bytes",
    "buffer_bytes",
    "tile_products",
    "ptx",
    "ttgir",
    "stages",
}
_PTX_FIELDS = {
    "version",
    "target",
    "instructions",
    "classes",
    "mma_forms",
    "ldmatrix_forms",
    "tensor_core_classes",
}
_TTGIR_FIELDS = {
    "target",
    "num_warps",
    "mma_versions",
    "mma",
    "shared",
    "dot_operands",
    "dots",
    "buffers",
    "ops",
}
_ENCODING_FIELDS = {"alias", "kind", "parameters"}
_OPERAND_FIELDS = {"alias", "kind", "parameters", "parent_kind"}
_DOT_FIELDS = {"op", "line", "result_encoding", "input_precision", "result_kind"}
_BUFFER_FIELDS = {
    "line",
    "shape",
    "element",
    "element_bits",
    "encoding",
    "encoding_kind",
    "memory",
    "bytes",
}
_STAGE_FIELDS = {"file", "sha256", "lines", "bytes"}
_INDEX_FIELDS = {"kernel_sha256", "config", "environment", "compiles"}
_INDEX_COMPILE_FIELDS = {
    "variant",
    "target",
    "shared_bytes",
    "num_warps",
    "num_stages",
    "stages",
}
_STAGES = 3
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_HEX = re.compile(r"[0-9a-f]{64}")


class TritonTcVerificationError(TritonTcError):
    """Raised when a Day 82 report does not reproduce from its inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonTcVerificationError(message)


def _same(expected: object, actual: object, path: str) -> None:
    try:
        _compare(expected, actual, path)
    except KernelLaunchVerificationError as error:
        raise TritonTcVerificationError(str(error)) from error


def _fields(value: object, expected: set[str], label: str) -> dict:
    _require(
        isinstance(value, dict) and set(value) == expected,
        f"{label} fields do not match",
    )
    return value


def _counts(value: object, names: tuple[str, ...]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(names)
        and all(type(item) is int and item >= 0 for item in value.values())
    )


def _read(path: Path, limit: int, label: str) -> bytes:
    try:
        return read_bounded(path, limit=limit, label=label)
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error


def _decimal(value: Fraction) -> str:
    """Round to 6 decimals, half to even, with integer arithmetic only."""

    sign = "-" if value < 0 else ""
    scaled = abs(value) * 10**6
    whole, rest = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rest
    if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
        whole += 1
    integer, part = divmod(whole, 10**6)
    return f"{sign if whole else ''}{integer}.{part:06d}"


def _ratio(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": _decimal(value),
    }


def _places(text: str) -> int:
    return len(text.partition(".")[2])


def _expected_device(device: Mapping, config: Mapping) -> dict[str, object]:
    factor = config["sparsity"]["factor"]
    sms = device["sm_count"]["value"]
    clock = device["boost_clock_mhz"]["value"]
    bandwidth = Fraction(device["memory_bandwidth_gb_per_second"]["value"])
    paths = device["paths"]
    rates = {name: paths[name]["fma_per_sm_per_clock"]["value"] for name in _PATHS}
    divisor = {
        name: factor if paths[name]["listed_tflops"]["with_sparsity"] else 1
        for name in _PATHS
    }
    dense = {
        name: Fraction(paths[name]["listed_tflops"]["value"]) / divisor[name]
        for name in _PATHS
    }
    reproduction = {}
    for name in _PATHS:
        if clock is None:
            reproduction[name] = None
            continue
        derived = 2 * rates[name] * sms * clock * 10**6
        published = dense[name] * 10**12
        text = paths[name]["listed_tflops"]["value"]
        half = Fraction(5 * 10 ** (11 - _places(text)), divisor[name])
        reproduction[name] = {
            "derived_flops_per_second": derived,
            "published_flops_per_second": int(published),
            "half_unit_flops_per_second": int(half),
            "within_rounding": abs(derived - published) <= half,
        }
    rows = []
    for size in config["sizes"]:
        points, times = {}, {}
        for name in _PATHS:
            flops = 2 * size**3
            moved = size * size * (2 * _INPUT[name] + config["output_bytes"])
            compute = Fraction(flops) / (dense[name] * 10**6)
            memory = Fraction(moved) / (bandwidth * 1000)
            times[name] = max(compute, memory)
            if memory > compute:
                bound = "memory"
            elif memory < compute:
                bound = "compute"
            else:
                bound = "ridge"
            points[name] = {
                "path": name,
                "flops": flops,
                "bytes": moved,
                "intensity": _ratio(Fraction(flops, moved)),
                "ridge": _ratio(dense[name] * 1000 / bandwidth),
                "compute_us": _ratio(compute),
                "memory_us": _ratio(memory),
                "time_us": _ratio(times[name]),
                "bound": bound,
            }
        rows.append(
            {
                "size": size,
                "points": points,
                "ratio": {name: _ratio(times[name] / times["bf16"]) for name in _PATHS},
                "memory_bound": [
                    name for name in _PATHS if points[name]["bound"] == "memory"
                ],
            }
        )
    return {
        "key": device["key"],
        "name": device["name"],
        "target": device["target"],
        "sm_count": sms,
        "bandwidth_gbps": _ratio(bandwidth),
        "fma_per_sm_per_clock": rates,
        "dense_tflops": {name: _ratio(dense[name]) for name in _PATHS},
        "per_clock_ratio": {
            name: _ratio(Fraction(rates["bf16"], rates[name])) for name in _PATHS
        },
        "peak_ratio": {name: _ratio(dense["bf16"] / dense[name]) for name in _PATHS},
        "implied_clock_mhz": {
            name: _ratio(dense[name] * 10**6 / (2 * rates[name] * sms))
            for name in _PATHS
        },
        "reproduction": reproduction,
        "ridge": {name: _ratio(dense[name] * 1000 / bandwidth) for name in _PATHS},
        "rows": rows,
    }


def expected_model(config: Mapping) -> dict[str, object]:
    """Rebuild the model section from the raw config mapping alone."""

    return {
        "label": MODEL_TEXT_LABEL,
        "rule": list(RULE),
        "omitted": list(OMITTED),
        "sizes": list(config["sizes"]),
        "devices": [_expected_device(device, config) for device in config["devices"]],
    }


def expected_kernel() -> dict[str, object]:
    """Rebuild the kernel section from a table written here, field by field."""

    rows = (
        ("fp32-ieee", "fp32", "ieee", 4, False, "fp32"),
        ("fp32-tf32", "fp32", "tf32", 4, True, "tf32"),
        ("fp16", "fp16", "ieee", 2, True, "fp16"),
        ("bf16", "bf16", "ieee", 2, True, "bf16"),
    )
    return {
        "language": "Triton (Python)",
        "kernel": "tc_matmul_kernel",
        "plan_file": "plans/d82/tc_kernel.py",
        "sha256": _KERNEL_SHA256,
        "text": TRITON_SOURCE,
        "config": {
            "name": "m128_n128_k32_w4_s3",
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
        },
        "variants": [
            {
                "name": name,
                "input_type": kind,
                "output_type": "fp32",
                "accumulator": "float32",
                "input_precision": precision,
                "element_bytes": size,
                "tensor_cores": tensor,
                "math": math_path,
                "mma_operand_type": _OPERAND[name],
            }
            for name, kind, precision, size, tensor, math_path in rows
        ],
    }


def _allowed(variant: str, target: str) -> list[str]:
    if variant == "fp32-ieee":
        return ["scalar_fma"]
    return ["mma_sync"] if target == "sm_80" else ["wgmma", "mma_sync"]


def _observed(classes: Mapping[str, int]) -> str:
    sync = classes["mma_sync"] > 0
    group = classes["wgmma_mma_async"] > 0
    if sync and group:
        return "mixed"
    if sync or group:
        return "mma_sync" if sync else "wgmma"
    return "scalar_fma" if classes["fma_rn_f32"] > 0 else "none"


def _form(opcode: object, label: str) -> dict[str, object]:
    """Read one MMA opcode with the grammar written here; return every field."""

    _require(type(opcode) is str, f"{label} must name its opcode")
    match = _MMA.fullmatch(opcode)
    if match:
        m, n, k, left, right, d, a, b, c = match.groups()
        family, layouts = "mma_sync", [left, right]
    else:
        match = _WGMMA.fullmatch(opcode)
        _require(match is not None, f"{label}: {opcode} is not an MMA of the grammar")
        (m, n, k, d, a, b), c = match.groups(), None
        family, layouts = "wgmma_mma_async", []
    m, n, k = int(m), int(n), int(k)
    return {
        "opcode": opcode,
        "family": family,
        "shape": f"m{m}n{n}k{k}",
        "m": m,
        "n": n,
        "k": k,
        "multiply_adds": m * n * k,
        "d_type": d,
        "a_type": a,
        "b_type": b,
        "c_type": c,
        "layouts": layouts,
    }


def _bits(element: object) -> int:
    _require(type(element) is str, "a buffer element must be a type name")
    if element in _BITS:
        return _BITS[element]
    match = re.fullmatch(r"[fiu](\d+)|f8\w+", element)
    _require(match is not None, f"buffer element {element} has no known width")
    return int(match.group(1)) if match.group(1) else 8


def _buffer_bytes(buffer: Mapping) -> int:
    product = 1
    for size in buffer["shape"]:
        product *= size
    return product * _bits(buffer["element"]) // 8


def _check_ptx_record(ptx: object, label: str) -> None:
    """Hold the PTX record of a row to its fields and to its own counts."""

    _fields(ptx, _PTX_FIELDS, label)
    _require(
        type(ptx["version"]) is str and type(ptx["target"]) is str,
        f"{label} must name its version and target",
    )
    classes = ptx["classes"]
    _require(_counts(classes, _CLASSES), f"{label}.classes must count each class")
    loads = ptx["ldmatrix_forms"]
    _require(
        isinstance(loads, dict)
        and all(
            type(name) is str
            and name.startswith("ldmatrix.sync.aligned.")
            and type(count) is int
            and count > 0
            for name, count in loads.items()
        )
        and sum(loads.values()) == classes["ldmatrix"],
        f"{label}.ldmatrix_forms must add up to its ldmatrix class",
    )
    _require(
        type(ptx["instructions"]) is int
        and ptx["instructions"] >= sum(classes.values()),
        f"{label}.instructions must hold every counted class",
    )
    forms = ptx["mma_forms"]
    _require(type(forms) is list, f"{label}.mma_forms must be a list")
    opcodes = [form.get("opcode") if isinstance(form, dict) else None for form in forms]
    _require(
        all(type(opcode) is str for opcode in opcodes)
        and opcodes == sorted(set(opcodes)),
        f"{label}.mma_forms must name each opcode once, in order",
    )
    for number, form in enumerate(forms):
        path = f"{label}.mma_forms[{number}]"
        count = form.get("count")
        _require(type(count) is int and count > 0, f"{path} must count its opcode")
        _same({**_form(form["opcode"], path), "count": count}, form, path)


def _check_ttgir_record(ttgir: object, label: str) -> None:
    """Hold the TTGIR record of a row to its fields and to its own counts."""

    _fields(ttgir, _TTGIR_FIELDS, label)
    _require(
        type(ttgir["target"]) is str and type(ttgir["num_warps"]) is int,
        f"{label} must name its target and warps",
    )
    for name, kinds in (("mma", _MMA_KINDS), ("shared", _SHARED_KINDS)):
        _require(type(ttgir[name]) is list, f"{label}.{name} must be a list")
        for number, item in enumerate(ttgir[name]):
            path = f"{label}.{name}[{number}]"
            _fields(item, _ENCODING_FIELDS, path)
            _require(
                type(item["alias"]) is str
                and item["kind"] in kinds
                and isinstance(item["parameters"], dict),
                f"{path} must be a {name} encoding alias",
            )
    versions = [item["parameters"].get("versionMajor", 0) for item in ttgir["mma"]]
    _require(
        all(type(version) is int for version in versions)
        and ttgir["mma_versions"] == sorted(set(versions)),
        f"{label}.mma_versions must be the versions of its mma encodings",
    )
    lists = ("dot_operands", "dots", "buffers")
    _require(
        all(type(ttgir[name]) is list for name in lists),
        f"{label} must hold lists of dot operands, dots and buffers",
    )
    for number, item in enumerate(ttgir["dot_operands"]):
        path = f"{label}.dot_operands[{number}]"
        _fields(item, _OPERAND_FIELDS, path)
        _require(
            item["alias"] is None
            and item["kind"] == "dot_op"
            and isinstance(item["parameters"], dict),
            f"{path} must be an inline dot_op encoding",
        )
    for number, item in enumerate(ttgir["dots"]):
        path = f"{label}.dots[{number}]"
        _fields(item, _DOT_FIELDS, path)
        _require(
            item["op"] in _DOT_OPS and type(item["line"]) is int and item["line"] > 0,
            f"{path} must name a dot op and its line",
        )
    for number, item in enumerate(ttgir["buffers"]):
        path = f"{label}.buffers[{number}]"
        _fields(item, _BUFFER_FIELDS, path)
        _require(
            type(item["line"]) is int
            and item["line"] > 0
            and type(item["shape"]) is list
            and all(type(size) is int and size > 0 for size in item["shape"]),
            f"{path} must name its line and a static shape",
        )
        _require(
            item["element_bits"] == _bits(item["element"]),
            f"{path}.element_bits does not match its element",
        )
    ops = ttgir["ops"]
    _require(_counts(ops, _TTGIR_OPS), f"{label}.ops must count each op")
    _require(
        ops["tt.dot"] + ops["ttng.warp_group_dot"] == len(ttgir["dots"]),
        f"{label}.dots must hold one row for each dot op",
    )
    _require(
        ops["ttg.local_alloc"] == len(ttgir["buffers"]),
        f"{label}.buffers must hold one row for each ttg.local_alloc",
    )


def _check_row(row: Mapping, variant: str, target: str) -> None:
    """Recompute one row's verdict from the counts that the row records."""

    label = f"compile.rows[{variant}@{target}]"
    _fields(row, _ROW_FIELDS, label)
    _require((row["variant"], row["target"]) == (variant, target), f"{label} order")
    ptx, ttgir = row["ptx"], row["ttgir"]
    _check_ptx_record(ptx, f"{label}.ptx")
    _check_ttgir_record(ttgir, f"{label}.ttgir")
    classes = ptx["classes"]
    issued = {"mma_sync": 0, "wgmma_mma_async": 0}
    for form in ptx["mma_forms"]:
        issued[form["family"]] += form["count"] * form["multiply_adds"]
    for family in issued:
        _require(
            classes[family]
            == sum(
                item["count"] for item in ptx["mma_forms"] if item["family"] == family
            ),
            f"{label}.ptx.classes does not count its MMA forms",
        )
    products = {
        "fma_rn_f32": Fraction(classes["fma_rn_f32"] * _THREADS, _TILE),
        "mma_sync": Fraction(issued["mma_sync"] * _WARPS, _TILE),
        "wgmma_mma_async": Fraction(issued["wgmma_mma_async"] * _GROUPS, _TILE),
    }
    allowed = _allowed(variant, target)
    observed = _observed(classes)
    key = {
        "scalar_fma": "fma_rn_f32",
        "mma_sync": "mma_sync",
        "wgmma": "wgmma_mma_async",
    }
    versions = {"scalar_fma": [], "mma_sync": [2], "wgmma": [3]}
    wanted = _OPERAND[variant]
    sizes = [_buffer_bytes(buffer) for buffer in ttgir["buffers"]]
    for buffer, size in zip(ttgir["buffers"], sizes):
        _require(buffer["bytes"] == size, f"{label}.ttgir.buffers bytes")
    whole = observed in key and products[key[observed]] > 0
    whole = whole and products[key[observed]].denominator == 1
    checks = {
        "target": ptx["target"].startswith(target)
        and ttgir["target"] == "cuda:" + target.removeprefix("sm_"),
        "class": observed in allowed,
        "operand_types": all(
            (form["a_type"], form["b_type"], form["d_type"]) == (wanted, wanted, "f32")
            for form in ptx["mma_forms"]
        )
        and (wanted is None) == (len(ptx["mma_forms"]) == 0),
        "mma_encoding": observed in versions
        and ttgir["mma_versions"] == versions[observed],
        "tile_products": whole,
        "no_scalar_dot": variant == "fp32-ieee" or products["fma_rn_f32"] < 1,
        "shared_fits": max(sizes, default=0) <= row["shared_bytes"],
    }
    note = None
    if target == "sm_90" and variant != "fp32-ieee" and observed == "mma_sync":
        note = "triton chose mma.sync on sm_90; the rule accepts and records it"
    _same(
        {
            "expected": allowed[0],
            "allowed": allowed,
            "observed": observed,
            "checks": checks,
            "accepted": all(checks.values()),
            "note": note,
            "largest_buffer_bytes": max(sizes, default=0),
            "buffer_bytes": sum(sizes),
            "tile_products": {
                name: {"numerator": value.numerator, "denominator": value.denominator}
                for name, value in products.items()
            },
            "ptx.tensor_core_classes": [
                name for name in ("mma_sync", "wgmma_mma_async") if classes[name]
            ],
        },
        {
            "expected": row["expected"],
            "allowed": row["allowed"],
            "observed": row["observed"],
            "checks": row["checks"],
            "accepted": row["accepted"],
            "note": row["note"],
            "largest_buffer_bytes": row["largest_buffer_bytes"],
            "buffer_bytes": row["buffer_bytes"],
            "tile_products": row["tile_products"],
            "ptx.tensor_core_classes": ptx["tensor_core_classes"],
        },
        label,
    )
    _require(
        type(row["shared_bytes"]) is int and 0 <= row["shared_bytes"] <= 227 * 1024,
        f"{label}.shared_bytes is out of range",
    )
    stages = _fields(row["stages"], {"ttgir", "ptx"}, f"{label}.stages")
    for stage in ("ttgir", "ptx"):
        record = _fields(stages[stage], _STAGE_FIELDS, f"{label}.stages.{stage}")
        _require(
            record["file"]
            == f"{variant}/{target}/"
            + ("ttgir.mlir" if stage == "ttgir" else "ptx.ptx")
            and type(record["sha256"]) is str
            and _HEX.fullmatch(record["sha256"]) is not None,
            f"{label}.stages.{stage} does not name its dump file and digest",
        )
        _require(
            all(
                type(record[name]) is int and record[name] > 0
                for name in ("lines", "bytes")
            ),
            f"{label}.stages.{stage} must name the lines and bytes of its text",
        )


def recount_ptx(text: str) -> dict[str, object]:
    """Rebuild the whole PTX record of a row with whole-text expressions."""

    cleaned = _PTX_TOKEN.sub(
        lambda found: '""'
        if found.group().startswith('"')
        else " " + "\n" * found.group().count("\n"),
        text,
    )
    _require(
        "/*" not in cleaned and "*/" not in cleaned,
        "a PTX text holds an unpaired block comment mark",
    )
    opcodes = Counter(_STATEMENT.findall(cleaned))
    targets = re.findall(r"^[ \t]*\.target[ \t]+(\S+)[ \t]*$", cleaned, re.M)
    versions = re.findall(r"^[ \t]*\.version[ \t]+(\S+)[ \t]*$", cleaned, re.M)
    _require(len(targets) == 1 and len(versions) == 1, "a PTX text needs one header")
    classes = dict.fromkeys(_CLASSES, 0)
    forms, loads = {}, {}
    for opcode, count in opcodes.items():
        name = _CLASS_EXACT.get(opcode)
        if name is None:
            name = next(
                (item for prefix, item in _CLASS_PREFIXES if opcode.startswith(prefix)),
                None,
            )
        if name is None and opcode.startswith("cvt.") and opcode.endswith(".tf32.f32"):
            name = "cvt_tf32"
        if name is None:
            continue
        classes[name] += count
        if name in ("mma_sync", "wgmma_mma_async"):
            forms[opcode] = count
        if name == "ldmatrix":
            loads[opcode] = count
    return {
        "version": versions[0],
        "target": targets[0],
        "instructions": sum(opcodes.values()),
        "classes": classes,
        "mma_forms": [
            {**_form(opcode, "ptx.mma_forms"), "count": count}
            for opcode, count in sorted(forms.items())
        ],
        "ldmatrix_forms": dict(sorted(loads.items())),
        "tensor_core_classes": [name for name in _FAMILIES if classes[name]],
    }


def _split(text: str) -> list[str]:
    """Split at each comma outside brackets; drop empty parts."""

    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<{([":
            depth += 1
        elif char in ">})]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _parameter(text: str) -> object:
    text = text.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if text in ("true", "false"):
        return text == "true"
    if text[:1] == "[" and text[-1:] == "]":
        return [_parameter(part) for part in _split(text[1:-1])]
    return text.strip('"')


def _parameters(body: str) -> dict[str, object]:
    """Parse the ``name = value`` parts of one encoding; keep the other parts."""

    found: dict[str, object] = {}
    others = []
    for part in _split(body):
        named = re.fullmatch(r"([A-Za-z_]\w*)\s*=(.*)", part, re.S)
        if named:
            found[named.group(1)] = _parameter(named.group(2))
        else:
            others.append(part)
    if others:
        found["unparsed"] = others
    return found


def recount_ttgir(text: str) -> dict[str, object]:
    """Rebuild the whole TTGIR record of a row from its text.

    Each line is an alias, the module line or a statement. The parser of the
    encoding parameters, the op grammar and the memdesc grammar are written
    here, not taken from the lab.
    """

    aliases: list[tuple[str, str, dict[str, object]]] = []
    modules, dots, buffers = [], [], []
    operands: dict[str, dict[str, object]] = {}
    ops = dict.fromkeys(_TTGIR_OPS, 0)
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        alias = re.fullmatch(
            r"#(\w+) = #(?:ttg|triton_gpu|ttng)\.(\w+)(?:<\{(.*)\}>)?", line
        )
        if alias:
            name, kind, body = alias.groups()
            aliases.append((name, kind, {} if body is None else _parameters(body)))
            continue
        module = re.fullmatch(r"module(?: attributes \{(.*)\})? \{", line)
        if module:
            modules.append(module.group(1) or "")
            continue
        if not line or line[0] in "#}^":
            continue
        for body in re.findall(r"#(?:ttg|triton_gpu)\.dot_op<\{([^{}]*)\}>", line):
            found = _parameters(body)
            operands.setdefault(repr(found), found)
        statement = re.match(
            r'(?:%[^=]*=[ \t]*)?"?([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+)', line
        )
        if statement is None:
            continue
        name = statement.group(1)
        if name in ops:
            ops[name] += 1
        if name in _DOT_OPS:
            result = re.search(r"->[ \t]*tensor<[^,>]*,[ \t]*(#\w+)>", line)
            precision = re.search(r"inputPrecision = (\w+)", line)
            dots.append(
                {
                    "op": name,
                    "line": number,
                    "result_encoding": result.group(1) if result else None,
                    "input_precision": precision.group(1) if precision else None,
                }
            )
        if name == "ttg.local_alloc":
            found = re.findall(
                r"!ttg\.memdesc<(\d+(?:x\d+)*)x(\w+), (#\w+)(?:, (#\w+))?", line
            )
            _require(bool(found), f"TTGIR line {number} allocates no memdesc")
            buffers.append((number, *found[-1]))
    _require(len(modules) == 1, "a TTGIR text needs one module")
    targets = re.findall(r'(?:^|, )"?ttg\.target"? = "([^"]*)"', modules[0])
    warps = re.findall(r'(?:^|, )"ttg\.num-warps" = (\d+) : i32', modules[0])
    _require(
        len(targets) == 1 and len(warps) == 1,
        "a TTGIR module needs one ttg.target and one ttg.num-warps",
    )
    kinds = {f"#{name}": kind for name, kind, _ in aliases}
    by_name = {}
    for name, kind, _ in aliases:
        by_name.setdefault(name, kind)
    mma = [
        {"alias": name, "kind": kind, "parameters": found}
        for name, kind, found in aliases
        if kind in _MMA_KINDS
    ]
    rows = []
    for number, shape, element, encoding, memory in buffers:
        buffer = {
            "line": number,
            "shape": [int(size) for size in shape.split("x")],
            "element": element,
            "element_bits": _bits(element),
            "encoding": encoding,
            "encoding_kind": kinds.get(encoding),
            "memory": memory or None,
        }
        buffer["bytes"] = _buffer_bytes(buffer)
        rows.append(buffer)
    return {
        "target": targets[0],
        "num_warps": int(warps[0]),
        "mma_versions": sorted(
            {int(item["parameters"].get("versionMajor", 0)) for item in mma}
        ),
        "mma": mma,
        "shared": [
            {"alias": name, "kind": kind, "parameters": found}
            for name, kind, found in aliases
            if kind in _SHARED_KINDS
        ],
        "dot_operands": [
            {
                "alias": None,
                "kind": "dot_op",
                "parameters": found,
                "parent_kind": by_name.get(
                    str(found.get("parent", "")).removeprefix("#")
                ),
            }
            for found in operands.values()
        ],
        "dots": [
            {**dot, "result_kind": kinds.get(dot["result_encoding"])} for dot in dots
        ],
        "buffers": rows,
        "ops": ops,
    }


def _check_dump(rows: list, directory: Path, section: Mapping) -> None:
    """Hold the dump index to the report; rebuild every row from its texts."""

    try:
        index = loads_strict(
            _read(directory / "index.json", 256 * 1024, "dump index"),
            label="dump index",
        )
    except KernelLaunchError as error:
        raise TritonTcVerificationError(str(error)) from error
    _fields(index, _INDEX_FIELDS, "dump index")
    _require(
        index["kernel_sha256"] == _KERNEL_SHA256,
        "dump index kernel_sha256 is not the pinned kernel text",
    )
    _same(expected_kernel()["config"], index["config"], "dump index config")
    _same(section["environment"], index["environment"], "dump index environment")
    compiles = index["compiles"]
    _require(
        type(compiles) is list
        and [
            (item.get("variant"), item.get("target"))
            for item in compiles
            if isinstance(item, dict)
        ]
        == _ORDER,
        "the dump index does not hold the compile order",
    )
    for row, record in zip(rows, compiles):
        label = f"dump[{row['variant']}@{row['target']}]"
        _fields(record, _INDEX_COMPILE_FIELDS, label)
        _require(
            (record["num_warps"], record["num_stages"]) == (_WARPS, _STAGES),
            f"{label} num_warps and num_stages are not the declared config",
        )
        _require(record["shared_bytes"] == row["shared_bytes"], f"{label} shared")
        stages = _fields(record["stages"], {"ttgir", "ptx"}, f"{label}.stages")
        texts = {}
        for stage in ("ttgir", "ptx"):
            listed = stages[stage]
            _same(listed, row["stages"][stage], f"{label}.stages.{stage}")
            data = _read(directory / listed["file"], MAX_TEXT_BYTES, stage)
            _require(
                sha256(data).hexdigest() == listed["sha256"],
                f"{label}.{stage} does not match its digest",
            )
            try:
                texts[stage] = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TritonTcVerificationError(
                    f"{label}.{stage} is not UTF-8"
                ) from error
            _require(
                listed["lines"] == len(texts[stage].splitlines())
                and listed["bytes"] == len(data),
                f"{label}.{stage} lines and bytes do not match its text",
            )
        _same(recount_ptx(texts["ptx"]), row["ptx"], f"{label}.ptx")
        _same(recount_ttgir(texts["ttgir"]), row["ttgir"], f"{label}.ttgir")


def _check_compile(section: object, label: str, dump: Path | None) -> None:
    if label == MODEL_LABEL:
        _require(section is None, "a declared_model report holds no compile section")
        _require(dump is None, "a declared_model report has no dump folder to read")
        return
    _require(isinstance(section, dict), "a compile report needs a compile section")
    _require(
        set(section) == {"backend", "environment", "rows", "summary"},
        "compile fields do not match",
    )
    backend = "triton_compiler" if label == COMPILER_LABEL else "injected"
    _require(section["backend"] == backend, "compile.backend does not match the label")
    environment = section["environment"]
    _require(
        isinstance(environment, dict)
        and set(environment) == {"python", "system", "machine", "triton"}
        and all(type(value) is str for value in environment.values()),
        "compile.environment must name python, system, machine and triton",
    )
    rows = section["rows"]
    _require(type(rows) is list and len(rows) == len(_ORDER), "compile.rows count")
    for row, (variant, target) in zip(rows, _ORDER):
        _check_row(row, variant, target)
    _same(
        {
            "compiles": len(rows),
            "accepted": sum(row["accepted"] for row in rows),
            "all_accepted": all(row["accepted"] for row in rows),
            "observed": {
                f"{v}@{t}": row["observed"] for row, (v, t) in zip(rows, _ORDER)
            },
            "mma_sync_on_sm_90": [row["variant"] for row in rows if row["note"]],
        },
        section["summary"],
        "compile.summary",
    )
    if dump is not None:
        _check_dump(rows, dump, section)


def verify_report(
    report: Mapping[str, object], config: Path, dump: Path | None = None
) -> dict[str, object]:
    """Verify a sealed report against its config file and an optional dump."""

    _require(isinstance(report, Mapping), "report must be a mapping")
    _require(set(report) == _REPORT_FIELDS, "report fields do not match")
    body = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    _require(
        sha256(canonical_json_bytes(body)).hexdigest() == report[DIGEST_FIELD],
        "report digest does not match its contents",
    )
    _require(
        report["schema_version"] == SCHEMA_VERSION and report["kind"] == REPORT_KIND,
        "report kind or schema does not match",
    )
    label = report["label"]
    _require(label in LABELS, "report label is unknown")
    _same(dict(BOUNDARIES[label]), report["evidence_boundary"], "evidence_boundary")
    digest = sha256(TRITON_SOURCE.encode("ascii")).hexdigest()
    _require(digest == _KERNEL_SHA256, "TRITON_SOURCE is not the pinned kernel text")
    _same(expected_kernel(), report["kernel"], "kernel")
    payload = _read(config, 64 * 1024, "config")
    declared = json.loads(payload.decode("utf-8"))
    _same(declared, report["config"], "config")
    _require(
        report["config_sha256"] == sha256(payload).hexdigest(),
        "config_sha256 does not match the config file",
    )
    _same(expected_model(declared), report["model"], "model")
    _check_compile(report["compile"], label, dump)
    _require(
        type(report["recorded_at"]) is str
        and _TIME.fullmatch(report["recorded_at"]) is not None,
        "recorded_at must be a UTC timestamp",
    )
    load = report["host_load"]
    _require(
        isinstance(load, dict)
        and set(load) == {"before", "after"}
        and all(
            type(values) is list
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0
                for value in values
            )
            for values in load.values()
        ),
        "host_load must hold three finite load averages before and after",
    )
    devices = report["model"]["devices"]
    compiled = report["compile"]
    _same(
        {
            "kernel_sha256": _KERNEL_SHA256,
            "config_sha256": sha256(payload).hexdigest(),
            "per_clock_ratio": {d["key"]: d["per_clock_ratio"] for d in devices},
            "largest_size_ratio": {d["key"]: d["rows"][-1]["ratio"] for d in devices},
            "memory_bound": {
                d["key"]: {
                    str(row["size"]): row["memory_bound"]
                    for row in d["rows"]
                    if row["memory_bound"]
                }
                for d in devices
            },
            "compile": None if compiled is None else compiled["summary"],
        },
        report["summary"],
        "summary",
    )
    return dict(report)


def load_report(
    path: Path, config: Path, dump: Path | None = None
) -> dict[str, object]:
    """Read one bounded report file and verify it."""

    payload = _read(path, MAX_REPORT_BYTES, "report")
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error
    _require(isinstance(report, dict), "report must be a JSON object")
    return verify_report(report, config, dump)
