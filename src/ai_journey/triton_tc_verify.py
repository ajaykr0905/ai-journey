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
* with a dump folder, every text by its digest, and the PTX and TTGIR
  counts with whole-text regular expressions.

Without a dump folder, the verifier cannot recount the texts, so it checks
only that each row agrees with its own counts. The verifier imports
neither triton nor torch.
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
_STATEMENT = re.compile(
    r"(?:^|[;{}\n])[ \t]*(?:[A-Za-z_$][\w$]*:[ \t]*)*(?:@!?%?\w+[ \t]+)?"
    r"([a-z][\w.:]*)"
)
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


def _form(opcode: str) -> dict[str, object]:
    """Read one MMA opcode with the grammar written here."""

    match = _MMA.fullmatch(opcode)
    if match:
        m, n, k, left, right, d, a, b, c = match.groups()
        family, layouts = "mma_sync", [left, right]
    else:
        match = _WGMMA.fullmatch(opcode)
        _require(match is not None, f"{opcode} is not an MMA opcode of the grammar")
        (m, n, k, d, a, b), c = match.groups(), None
        family, layouts = "wgmma_mma_async", []
    m, n, k = int(m), int(n), int(k)
    return {
        "family": family,
        "shape": f"m{m}n{n}k{k}",
        "multiply_adds": m * n * k,
        "d_type": d,
        "a_type": a,
        "b_type": b,
        "c_type": c,
        "layouts": layouts,
    }


def _buffer_bytes(buffer: Mapping) -> int:
    product = 1
    for size in buffer["shape"]:
        product *= size
    element = buffer["element"]
    bits = _BITS.get(element)
    if bits is None:
        match = re.fullmatch(r"[fiu](\d+)|f8\w+", element)
        _require(match is not None, f"buffer element {element} has no known width")
        bits = int(match.group(1)) if match.group(1) else 8
    return product * bits // 8


def _check_row(row: Mapping, variant: str, target: str) -> None:
    """Recompute one row's verdict from the counts that the row records."""

    label = f"compile.rows[{variant}@{target}]"
    _require((row["variant"], row["target"]) == (variant, target), f"{label} order")
    ptx, ttgir = row["ptx"], row["ttgir"]
    classes = ptx["classes"]
    issued = {"mma_sync": 0, "wgmma_mma_async": 0}
    for form in ptx["mma_forms"]:
        read = _form(form["opcode"])
        _same(read, {key: form[key] for key in read}, f"{label}.mma_forms")
        issued[read["family"]] += form["count"] * read["multiply_adds"]
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
    for stage in ("ttgir", "ptx"):
        record = row["stages"][stage]
        _require(
            record["file"]
            == f"{variant}/{target}/"
            + ("ttgir.mlir" if stage == "ttgir" else "ptx.ptx")
            and type(record["sha256"]) is str
            and _HEX.fullmatch(record["sha256"]) is not None,
            f"{label}.stages.{stage} does not name its dump file and digest",
        )


def recount_ptx(text: str) -> dict[str, object]:
    """Recount the PTX classes and MMA forms with whole-text expressions."""

    cleaned = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", text, flags=re.S))
    opcodes = Counter(_STATEMENT.findall(cleaned))
    targets = re.findall(r"^[ \t]*\.target[ \t]+(\S+)[ \t]*$", cleaned, re.M)
    versions = re.findall(r"^[ \t]*\.version[ \t]+(\S+)[ \t]*$", cleaned, re.M)
    _require(len(targets) == 1 and len(versions) == 1, "a PTX text needs one header")
    classes = dict.fromkeys(
        (
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
        ),
        0,
    )
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
        "mma_forms": dict(sorted(forms.items())),
        "ldmatrix_forms": dict(sorted(loads.items())),
    }


def recount_ttgir(text: str) -> dict[str, object]:
    """Recount the MMA versions, swizzles and buffers with expressions."""

    aliases = dict(re.findall(r"^#(\w+) = #(?:ttg|triton_gpu|ttng)\.(\w+)", text, re.M))
    versions = sorted(
        {
            int(version)
            for kind, version in re.findall(
                r"^#\w+ = #(?:ttg|triton_gpu)\.(nvidia_mma|mma)<\{[^\n]*?"
                r"versionMajor = (\d+)",
                text,
                re.M,
            )
        }
    )
    targets = re.findall(r'ttg\.target = "([^"]+)"', text)
    _require(len(targets) == 1, "a TTGIR text needs one ttg.target")
    buffers = []
    for line in re.findall(r"^[^\n]*= ttg\.local_alloc[^\n]*$", text, re.M):
        found = re.findall(r"!ttg\.memdesc<([0-9x]+)x(\w+), (#\w+)", line)
        _require(bool(found), "a local_alloc line names no memdesc")
        shape, element, encoding = found[-1]
        buffer = {
            "shape": [int(item) for item in shape.split("x")],
            "element": element,
            "encoding_kind": aliases.get(encoding[1:]),
        }
        buffer["bytes"] = _buffer_bytes(buffer)
        buffers.append(buffer)
    swizzles = [
        dict(re.findall(r"(vec|perPhase|maxPhase|swizzlingByteWidth) = (\d+)", line))
        for line in re.findall(
            r"^#\w+ = #(?:ttg|triton_gpu)\.(?:swizzled_shared|shared|nvmma_shared|shared_linear)<"
            r"[^\n]*$",
            text,
            re.M,
        )
    ]
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    mma = [
        {
            "alias": alias,
            "versionMajor": int(re.search(r"versionMajor = (\d+)", rest).group(1)),
            "versionMinor": int(re.search(r"versionMinor = (\d+)", rest).group(1)),
            "warpsPerCTA": _integers(re.search(r"warpsPerCTA = \[([^\]]*)\]", rest)),
            "instrShape": _integers(re.search(r"instrShape = \[([^\]]*)\]", rest)),
        }
        for alias, rest in re.findall(
            r"^#(\w+) = #(?:ttg|triton_gpu)\.(?:nvidia_mma|mma)<\{([^\n]*)\}>$",
            text,
            re.M,
        )
    ]
    operands = []
    for item in re.findall(r"#(?:ttg|triton_gpu)\.dot_op<\{([^}]*)\}>", body):
        found = dict(re.findall(r"(opIdx|kWidth) = (\d+)", item))
        parent = re.search(r"parent = (#\w+)", item)
        operand = {key: int(value) for key, value in found.items()}
        operand["parent"] = parent.group(1) if parent else None
        if operand not in operands:
            operands.append(operand)
    ops = {
        name: len(
            re.findall(
                r'^[ \t]*(?:%[^=\n]*=[ \t]*)?"?'
                + re.escape(name)
                + r'"?(?=[ \t({<"]|$)',
                body,
                re.M,
            )
        )
        for name in _TTGIR_OPS
    }
    return {
        "target": targets[0],
        "mma_versions": versions,
        "buffers": buffers,
        "swizzles": [
            {key: int(value) for key, value in item.items()} for item in swizzles
        ],
        "mma": mma,
        "dot_operands": operands,
        "ops": ops,
    }


def _integers(match: re.Match | None) -> list[int] | None:
    if match is None:
        return None
    return [int(item) for item in match.group(1).split(",") if item.strip()]


def _check_dump(rows: list, directory: Path) -> None:
    """Recount every text of a dump folder and compare it with its row."""

    index = json.loads(_read(directory / "index.json", 256 * 1024, "dump index"))
    _require(
        [(item["variant"], item["target"]) for item in index["compiles"]] == _ORDER,
        "the dump index does not hold the compile order",
    )
    for row, record in zip(rows, index["compiles"]):
        label = f"dump[{row['variant']}@{row['target']}]"
        texts = {}
        for stage in ("ttgir", "ptx"):
            listed = record["stages"][stage]
            _same(listed, row["stages"][stage], f"{label}.stages.{stage}")
            data = _read(directory / listed["file"], MAX_TEXT_BYTES, stage)
            _require(
                sha256(data).hexdigest() == listed["sha256"],
                f"{label}.{stage} does not match its digest",
            )
            texts[stage] = data.decode("utf-8")
        _require(record["shared_bytes"] == row["shared_bytes"], f"{label} shared")
        ptx = recount_ptx(texts["ptx"])
        _same(
            ptx,
            {
                "version": row["ptx"]["version"],
                "target": row["ptx"]["target"],
                "instructions": row["ptx"]["instructions"],
                "classes": row["ptx"]["classes"],
                "mma_forms": {
                    form["opcode"]: form["count"] for form in row["ptx"]["mma_forms"]
                },
                "ldmatrix_forms": row["ptx"]["ldmatrix_forms"],
            },
            f"{label}.ptx",
        )
        ttgir = recount_ttgir(texts["ttgir"])
        _same(
            ttgir,
            {
                "target": row["ttgir"]["target"],
                "mma_versions": row["ttgir"]["mma_versions"],
                "buffers": [
                    {
                        "shape": item["shape"],
                        "element": item["element"],
                        "encoding_kind": item["encoding_kind"],
                        "bytes": item["bytes"],
                    }
                    for item in row["ttgir"]["buffers"]
                ],
                "swizzles": [
                    {
                        key: value
                        for key, value in item["parameters"].items()
                        if key in ("vec", "perPhase", "maxPhase", "swizzlingByteWidth")
                    }
                    for item in row["ttgir"]["shared"]
                ],
                "mma": [
                    {
                        "alias": item["alias"],
                        **{
                            key: item["parameters"].get(key)
                            for key in (
                                "versionMajor",
                                "versionMinor",
                                "warpsPerCTA",
                                "instrShape",
                            )
                        },
                    }
                    for item in row["ttgir"]["mma"]
                ],
                "dot_operands": [
                    {
                        **{
                            key: item["parameters"][key]
                            for key in ("opIdx", "kWidth")
                            if key in item["parameters"]
                        },
                        "parent": item["parameters"].get("parent"),
                    }
                    for item in row["ttgir"]["dot_operands"]
                ],
                "ops": row["ttgir"]["ops"],
            },
            f"{label}.ttgir",
        )


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
        _check_dump(rows, dump)


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
