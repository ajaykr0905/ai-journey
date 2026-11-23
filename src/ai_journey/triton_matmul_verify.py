"""Independent verification of sealed Triton matmul comparison reports, on the CPU.

Verification recomputes each number that the seed and the pinned texts fix,
and it checks the rest for structure. It checks the field set, the label and
its evidence boundary and the SHA-256 seal, parses the embedded plan, and can
bind the report to a plan file. It then rebuilds each section, with other code
than the run's where the run used lab code:

* the operands, the float64 reference and the bound of each case with the
  Day 74 Metal verifier's code, and the exact operands from the seed;
* the programs, the K blocks and the guard of each run by closed forms;
* each exact buffer from the float64 reference and its sentinel tail, so its
  digest is checked bit for bit on any CPU;
* the line counts with a second counter: ``//`` stripping line by line and
  brace spans for the C texts, and ``ast`` spans with a prefix test for the
  Triton text.

A normal output has an unknown order of additions, so its record is checked
for structure: a digest, an error within the bound, a ratio that recomputes,
and a count of elements off the rounded reference that is zero exactly when
the digest is the digest of the rounded reference. No float32 output is
nearer the float64 reference than its rounding, so no error may be below the
error of the rounded reference. A record with the digest of the rounded
reference holds its bits, so it must record its error, as the Day 74 verifier
requires of a PyTorch record. The Metal rates, the later GPU steps and the
PyPI facts are recorded data and must equal the pinned data. The summary is
rebuilt last. The verifier imports neither triton nor torch, so a macOS host
can verify a Linux interpreter report.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import numpy

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_verify import KernelLaunchVerificationError, _compare
from .matmul_metal_verify import (
    _bound,
    _hash,
    _operands,
    _reference,
    _relative_error,
)
from .triton_matmul_config import TritonMatmulPlan
from .triton_matmul_kernel import TRITON_SOURCE, TUTORIAL, TritonMatmulError
from .triton_matmul_loc import COUNTING_RULE, counted_texts
from .triton_matmul_performance import performance_mapping
from .triton_matmul_report import (
    DIGEST_FIELD,
    INJECTED_BACKEND,
    INTERPRETER_BOUNDARY,
    INTERPRETER_LABEL,
    REPORT_KIND,
    SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    rule_mapping,
)
from .triton_matmul_runtime import (
    ENVIRONMENT_FIELDS,
    INTERPRETER_BACKEND,
    ci_install_mapping,
)

MAX_REPORT_BYTES = 2 * 1024 * 1024
# Pinned here independently of the modules that the run uses.
_SENTINEL = 0x7FC00000
_EXACT_STREAM = 2
_EXACT_LIMIT = 8
_FLOAT = 4
_FAMILIES = ("normal", "exact")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_LABELS = {
    INTERPRETER_LABEL: (INTERPRETER_BACKEND, INTERPRETER_BOUNDARY),
    SYNTHETIC_LABEL: (INJECTED_BACKEND, SYNTHETIC_BOUNDARY),
}
_REPORT_FIELDS = {
    "schema_version",
    "kind",
    "label",
    "backend",
    "evidence_boundary",
    "plan",
    "plan_sha256",
    "environment",
    "recorded_at",
    "host_load",
    "kernel",
    "rule",
    "correctness",
    "loc",
    "performance",
    "ci_install",
    "summary",
    DIGEST_FIELD,
}


class TritonMatmulVerificationError(TritonMatmulError):
    """Raised when a report does not reproduce from its declared plan."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonMatmulVerificationError(message)


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _expected_kernel(plan: TritonMatmulPlan) -> dict[str, object]:
    configs = []
    for config in plan.configs:
        bm, bn, bk = config.block_m, config.block_n, config.block_k
        configs.append(
            {
                "name": f"m{bm}_n{bn}_k{bk}_g{config.group_m}_w{config.num_warps}"
                f"_s{config.num_stages}",
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "group_m": config.group_m,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "tile_bytes": (bm * bk + bk * bn) * _FLOAT,
            }
        )
    return {
        "language": "Triton (Python)",
        "kernel": "matmul_kernel",
        "launcher": "launch_matmul",
        "input_precision": "ieee",
        "accumulator": "float32",
        "sha256": sha256(TRITON_SOURCE.encode("ascii")).hexdigest(),
        "text": TRITON_SOURCE,
        "autotune": {
            "key": ["M", "N", "K"],
            "configs": configs,
            "tuned": False,
            "source": TUTORIAL,
        },
    }


def _span_end(code: list[str], start: int) -> int:
    """Return the index of the line where the braces that open at ``start`` close."""

    depth = 0
    for index in range(start, len(code)):
        depth += code[index].count("{") - code[index].count("}")
        if depth == 0 and "{" in "".join(code[start : index + 1]):
            return index
    raise TritonMatmulVerificationError("a counted text never closes a function")


def _recount_c(text: str, metal: bool) -> dict[str, int | None]:
    """Count a pinned C text by ``//`` stripping and kernel brace spans."""

    _require(
        not any(token in text for token in ('"', "/*", "\\", "??")),
        "a counted C text holds a string, a block comment or a splice",
    )
    lines = text.split("\n")[:-1]
    code = [line.split("//", 1)[0] for line in lines]
    blank = sum(not line.strip() for line in lines)
    comment = sum(
        bool(line.strip()) and not part.strip() for line, part in zip(lines, code)
    )
    other = sum(part.lstrip().startswith("#") for part in code)
    marker = "kernel void " if metal else "__global__ "
    kernel = 0
    for start, part in enumerate(code):
        if part.startswith(marker):
            end = _span_end(code, start)
            kernel += sum(bool(item.strip()) for item in code[start : end + 1])
    host = len(lines) - blank - comment - other - kernel
    if metal:
        _require(host == 0, "a counted Metal text holds code outside its kernels")
    return {
        "kernel": kernel,
        "host": None if metal else host,
        "other": other,
        "comment": comment,
        "blank": blank,
        "total": len(lines),
    }


def _recount_python(text: str) -> dict[str, int | None]:
    """Count the pinned Triton text by ``ast`` spans and a ``#`` prefix test."""

    _require(
        '"""' not in text and "'''" not in text, "the Triton text holds a docstring"
    )
    lines = text.split("\n")[:-1]
    kinds = ["blank" if not line.strip() else "comment" for line in lines]
    for node in ast.parse(text).body:
        decorators = [ast.unparse(item) for item in getattr(node, "decorator_list", [])]
        first = min(
            [
                node.lineno,
                *(item.lineno for item in getattr(node, "decorator_list", [])),
            ]
        )
        kind = (
            "kernel"
            if "triton.jit" in decorators
            else "host"
            if isinstance(node, ast.FunctionDef)
            else "other"
        )
        for number in range(first, node.end_lineno + 1):
            stripped = lines[number - 1].strip()
            if stripped and not stripped.startswith("#"):
                kinds[number - 1] = kind
    counts = {
        name: kinds.count(name)
        for name in ("kernel", "host", "other", "comment", "blank")
    }
    return {**counts, "total": len(lines)}


def _expected_loc() -> dict[str, object]:
    texts = []
    for item in counted_texts():
        if item.language == "Triton (Python)":
            lines = _recount_python(item.text)
        else:
            lines = _recount_c(item.text, item.language == "Metal Shading Language")
        texts.append(
            {
                "lab": item.lab,
                "language": item.language,
                "name": item.name,
                "origin": item.origin,
                "sha256": sha256(item.text.encode("ascii")).hexdigest(),
                "lines": lines,
            }
        )
    return {"rule": COUNTING_RULE, "texts": texts}


def _exact_operands(seed: int, index: int, m: int, n: int, k: int):
    generator = numpy.random.default_rng([seed, _EXACT_STREAM, index])
    left = generator.integers(-_EXACT_LIMIT, _EXACT_LIMIT + 1, (m, k))
    right = generator.integers(-_EXACT_LIMIT, _EXACT_LIMIT + 1, (k, n))
    return left.astype(numpy.float32), right.astype(numpy.float32)


def _normal_run(
    record: object,
    expected: dict[str, object],
    digest: str,
    size: int,
    bound: float,
    floor: float,
    path: str,
) -> None:
    """Check the structure of a normal record, then its fixed fields.

    ``digest`` and ``floor`` are the digest and the error of the rounded
    reference buffer. No float32 value is nearer a float64 sum than its
    rounding, so no output has a smaller error, and an output with that
    digest holds those bits, so its error is ``floor`` exactly.
    """

    _require(
        type(record) is dict and set(record) == set(expected),
        f"{path} fields do not match",
    )
    error, ratio = record["max_relative_error"], record["bound_ratio"]
    differences, buffer = (
        record["rounded_reference_differences"],
        record["buffer_sha256"],
    )
    _require(
        type(error) is float and math.isfinite(error) and 0 <= error <= bound,
        f"{path} has no error within the bound",
    )
    _require(
        type(ratio) is float and ratio == error / bound,
        f"{path} has a ratio that does not recompute",
    )
    _require(
        type(differences) is int and 0 <= differences <= size,
        f"{path} has no count of elements off the rounded reference",
    )
    _require(
        type(buffer) is str and _SHA256.fullmatch(buffer) is not None,
        f"{path} has no buffer digest",
    )
    _require(
        (differences == 0) == (buffer == digest),
        f"{path} has a difference count that disagrees with its digest",
    )
    _require(
        floor <= error,
        f"{path} has an error below the error of the rounded reference, which "
        "no float32 output can have",
    )
    _require(
        buffer != digest or error == floor,
        f"{path} holds the bits of the rounded reference, so it must record "
        "their error",
    )
    _compare(
        {
            **expected,
            "max_relative_error": error,
            "bound_ratio": ratio,
            "rounded_reference_differences": differences,
            "buffer_sha256": buffer,
        },
        record,
        path,
    )


def _verify_case(plan: TritonMatmulPlan, index: int, actual: object) -> list[dict]:
    """Rebuild one case and compare it; return its verified run records."""

    shape = plan.shapes[index]
    m, n, k = shape.m, shape.n, shape.k
    path = f"correctness.cases[{index}]"
    _require(
        type(actual) is dict and type(actual.get("runs")) is list, f"{path} has no runs"
    )
    bound = _bound(k)
    families = {
        "normal": _operands(plan.seed, index, m, n, k),
        "exact": _exact_operands(plan.seed, index, m, n, k),
    }
    operands, rounded, floor = [], {}, {}
    for family, (left, right) in families.items():
        total, scale = _reference(left, right)
        rounded[family] = total.astype(numpy.float32)
        floor[family] = _relative_error(rounded[family], total, scale)
        operands.append(
            {
                "family": family,
                "a_sha256": _hash(left),
                "b_sha256": _hash(right),
                "reference_sha256": _hash(total),
            }
        )
    records = actual["runs"]
    _require(
        len(records) == len(plan.configs) * len(_FAMILIES), f"{path} runs do not match"
    )
    runs = []
    for position, record in enumerate(records):
        config = plan.configs[position // len(_FAMILIES)]
        family = _FAMILIES[position % len(_FAMILIES)]
        bm, bn, bk = config.block_m, config.block_n, config.block_k
        guard = (_ceil(m, bm) * bm - 1) * n + _ceil(n, bn) * bn - m * n
        tail = numpy.full(guard, _SENTINEL, dtype=numpy.uint32).view(numpy.float32)
        digest = _hash(rounded[family].reshape(-1), tail)
        expected = {
            "config": f"m{bm}_n{bn}_k{bk}_g{config.group_m}_w{config.num_warps}"
            f"_s{config.num_stages}",
            "family": family,
            "programs": _ceil(m, bm) * _ceil(n, bn),
            "k_blocks": _ceil(k, bk),
            "guard_elements": guard,
            "buffer_sha256": digest,
            "max_relative_error": 0.0,
            "bound_ratio": 0.0,
            "rounded_reference_differences": 0,
        }
        where = f"{path}.runs[{position}]"
        if family == "exact":
            _compare(expected, record, where)
        else:
            _normal_run(record, expected, digest, m * n, bound, floor[family], where)
        runs.append(record)
    _compare(
        {
            "index": index,
            "shape": {"m": m, "n": n, "k": k},
            "error_bound": bound,
            "operands": operands,
            "runs": records,
        },
        actual,
        path,
    )
    return runs


def _expected_summary(
    plan: TritonMatmulPlan, runs: list[dict], loc: dict
) -> dict[str, object]:
    normal = [run for run in runs if run["family"] == "normal"]
    exact = [run for run in runs if run["family"] == "exact"]
    steps = 2 * sum(
        _ceil(shape.m, config.block_m)
        * _ceil(shape.n, config.block_n)
        * _ceil(shape.k, config.block_k)
        for config in plan.configs
        for shape in plan.shapes
    )
    triton = loc["texts"][0]["lines"]
    return {
        "cases": len(plan.shapes),
        "configs": len(plan.configs),
        "runs": len(runs),
        "normal_runs_within_bound": len(normal),
        "exact_runs_with_exact_bits": len(exact),
        "max_bound_ratio": max(run["bound_ratio"] for run in normal),
        "normal_elements_off_the_rounded_reference": sum(
            run["rounded_reference_differences"] for run in normal
        ),
        "interpreter_steps": steps,
        "loc_texts": len(loc["texts"]),
        "triton_kernel_lines": triton["kernel"],
        "triton_host_lines": triton["host"],
    }


def _environment(environment: object, host_load: object, recorded_at: object) -> None:
    _require(
        type(environment) is dict
        and set(environment) == set(ENVIRONMENT_FIELDS)
        and all(
            type(value) is str and _TOKEN.fullmatch(value)
            for value in environment.values()
        ),
        "environment must name each field with a short token",
    )
    _require(
        type(recorded_at) is str and _TIMESTAMP.fullmatch(recorded_at) is not None,
        "recorded_at must be a UTC time stamp",
    )
    _require(
        type(host_load) is dict
        and set(host_load) == {"before", "after"}
        and all(
            type(values) is list
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0
                for value in values
            )
            for values in host_load.values()
        ),
        "host_load must hold three load averages before and after",
    )


def _verify(
    report: Mapping[str, object], plan: TritonMatmulPlan | None
) -> TritonMatmulPlan:
    _require(
        type(report["schema_version"]) is int
        and report["schema_version"] == SCHEMA_VERSION,
        "unsupported report schema_version",
    )
    _require(report["kind"] == REPORT_KIND, f"report kind is not {REPORT_KIND}")
    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    _require(
        report[DIGEST_FIELD] == _canonical_digest(unsigned),
        "report digest does not match its content",
    )
    _require(
        type(report["label"]) is str and report["label"] in _LABELS,
        "report label is not a known label",
    )
    backend, boundary = _LABELS[report["label"]]
    _require(report["backend"] == backend, "report backend does not match its label")
    _compare(dict(boundary), report["evidence_boundary"], "evidence_boundary")
    try:
        embedded = TritonMatmulPlan.from_mapping(report["plan"])
    except (ValueError, TypeError) as error:
        raise TritonMatmulVerificationError(
            f"embedded plan is invalid: {error}"
        ) from error
    _compare(embedded.to_mapping(), report["plan"], "plan")
    if plan is not None:
        if type(plan) is not TritonMatmulPlan:
            raise TypeError("plan must be TritonMatmulPlan")
        _require(plan == embedded, "report was not built from this plan")
    _require(
        report["plan_sha256"] == _canonical_digest(report["plan"]),
        "plan_sha256 does not match the plan",
    )
    _environment(report["environment"], report["host_load"], report["recorded_at"])
    loc = _expected_loc()
    static = {
        "kernel": _expected_kernel(embedded),
        "rule": {**rule_mapping(), "sentinel_float32_bits": _SENTINEL},
        "loc": loc,
        "performance": performance_mapping(),
        "ci_install": ci_install_mapping(),
    }
    for name, section in static.items():
        _compare(section, report[name], name)
    correctness = report["correctness"]
    _require(
        type(correctness) is dict
        and set(correctness) == {"cases"}
        and type(correctness["cases"]) is list
        and len(correctness["cases"]) == len(embedded.shapes),
        "correctness cases do not match the plan",
    )
    runs = []
    for index, actual in enumerate(correctness["cases"]):
        runs += _verify_case(embedded, index, actual)
    _compare(_expected_summary(embedded, runs, loc), report["summary"], "summary")
    return embedded


def verify_report(
    report: object, *, plan: TritonMatmulPlan | None = None
) -> TritonMatmulPlan:
    """Return the report's plan only if every field reproduces exactly."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise TritonMatmulVerificationError("report fields do not match the schema")
    try:
        return _verify(report, plan)
    except KernelLaunchVerificationError as error:
        # The Day 73 comparison helper names the field; keep its message.
        raise TritonMatmulVerificationError(str(error)) from error


def load_report(
    path: Path, *, plan: TritonMatmulPlan | None = None
) -> dict[str, object]:
    """Load canonical report bytes and verify every field.

    A file that cannot be read raises ``TritonMatmulError``. A file that was
    read but is not a canonical, reproducible report raises
    ``TritonMatmulVerificationError``.
    """

    try:
        payload = read_bounded(path, limit=MAX_REPORT_BYTES, label="report")
    except KernelLaunchError as error:
        raise TritonMatmulError(str(error)) from error
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise TritonMatmulVerificationError(str(error)) from error
    _require(type(report) is dict, "report must be a JSON object")
    _require(
        payload == canonical_json_bytes(report) + b"\n",
        "report bytes are not canonical JSON",
    )
    verify_report(report, plan=plan)
    return report
