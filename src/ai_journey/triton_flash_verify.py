"""Independent verification of sealed Day 81 flash attention reports, on the CPU.

Verification recomputes each number that the seed, the plan and the stored
values fix, with other code than the run's:

* the Q, K and V of each case from the seed, hashed with ``hashlib``;
* the programs, key blocks and guard rows of each run by closed forms;
* each stored O, m and l from its base64 text, with its digest, its shape, its
  finiteness and the rule ``l >= 1``;
* a float64 reference by a second formula, ``P = exp(S - LSE)``, and from it
  each output error and log-sum-exp error;
* each declared bound with its own loop over the heads;
* the memory and traffic table from the closed forms of the rule.

The float64 reference and the bounds go through ``numpy.matmul``, whose order
of additions depends on the BLAS build, so a recomputed error or bound must
match its record within a relative tolerance of 1e-6, not bit for bit. Every
recomputed ratio must be at most 1, whatever the record says. The verifier
imports neither triton nor torch, so a macOS host can verify a Linux report.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import numpy

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, loads_strict, read_bounded
from .kernel_launch_verify import KernelLaunchVerificationError, _compare
from .triton_flash_config import TritonFlashPlan
from .triton_flash_kernel import TRITON_SOURCE, TritonFlashError
from .triton_flash_report import (
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
from .triton_flash_runtime import INTERPRETER_BACKEND
from .triton_matmul_runtime import ci_install_mapping
from .triton_matmul_verify import TritonMatmulVerificationError
from .triton_matmul_verify import _environment as _matmul_environment

MAX_REPORT_BYTES = 4 * 1024 * 1024
# Pinned here independently of the modules that the run uses.
_SENTINEL = 0x7FC00000
_U = 2.0**-24
_STREAMS = {"normal": 0, "large": 1}
_LARGE = 8.0
_FLOAT = 4
_DIMS = (16, 32, 64, 128)
_TOLERANCE = 1e-6
_FLOOR = 1e-12
_SEQUENCES = (512, 1024, 2048, 4096, 8192)
_MODEL_DIMS = (64, 128)
_TRAFFIC_BLOCK_M = 128
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
    "memory",
    "ci_install",
    "summary",
    DIGEST_FIELD,
}
_RUN_FIELDS = {
    "config",
    "family",
    "programs",
    "key_blocks",
    "guard_rows",
    "output_sha256",
    "maxima_sha256",
    "sums_sha256",
    "output",
    "maxima",
    "sums",
    "output_bounds",
    "lse_bounds",
    "max_output_error",
    "output_bound_ratio",
    "max_lse_error",
    "lse_bound_ratio",
}


class TritonFlashVerificationError(TritonFlashError):
    """Raised when a report does not reproduce from its declared plan."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonFlashVerificationError(message)


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


def _scale(dim: int) -> float:
    return float(numpy.float32(1.0 / math.sqrt(dim)))


def _close(recorded: object, recomputed: float, label: str) -> None:
    _require(
        type(recorded) is float
        and math.isfinite(recorded)
        and abs(recorded - recomputed) <= _TOLERANCE * abs(recomputed) + _FLOOR,
        f"{label} does not match independent recomputation",
    )


def _expected_kernel(plan: TritonFlashPlan) -> dict[str, object]:
    configs = []
    for config in plan.configs:
        bm, bn = config.block_m, config.block_n
        configs.append(
            {
                "name": f"m{bm}_n{bn}_w{config.num_warps}_s{config.num_stages}",
                "block_m": bm,
                "block_n": bn,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "tile_bytes": {
                    str(d): (bm * d + 2 * bn * d + bm * bn) * _FLOAT for d in _DIMS
                },
            }
        )
    return {
        "language": "Triton (Python)",
        "kernel": "flash_attention_forward",
        "launcher": "launch_flash_attention",
        "input_precision": "ieee",
        "accumulator": "float32",
        "grid": "(cdiv(N, BLOCK_M), BH)",
        "sha256": sha256(TRITON_SOURCE.encode("ascii")).hexdigest(),
        "text": TRITON_SOURCE,
        "configs": configs,
        "tuned": False,
    }


def _expected_memory(memory: object) -> None:
    _require(
        isinstance(memory, dict) and type(memory.get("rows")) is list,
        "memory must hold rows",
    )
    rows = []
    for d in _MODEL_DIMS:
        for n in _SEQUENCES:
            io = 4 * n * d * _FLOAT
            naive = (2 * n * n) * _FLOAT + io
            flash = 2 * n * _FLOAT + io
            naive_traffic = (4 * n * n + 4 * n * d) * _FLOAT
            flash_traffic = (
                2 * n * d + 2 * n * d * _ceil(n, _TRAFFIC_BLOCK_M) + 2 * n
            ) * _FLOAT
            rows.append(
                {
                    "sequence": n,
                    "head_dim": d,
                    "io_bytes": io,
                    "naive_intermediate_bytes": 2 * n * n * _FLOAT,
                    "naive_bytes": naive,
                    "flash_intermediate_bytes": 2 * n * _FLOAT,
                    "flash_bytes": flash,
                    "memory_ratio": naive / flash,
                    "naive_traffic_bytes": naive_traffic,
                    "flash_traffic_bytes": flash_traffic,
                    "traffic_ratio": naive_traffic / flash_traffic,
                }
            )
    _compare(rows, memory["rows"], "memory.rows")
    _require(
        memory.get("measured") is False
        and memory.get("traffic_block_m") == _TRAFFIC_BLOCK_M,
        "memory must be an unmeasured model with BLOCK_M 128",
    )
    for field in ("unit", "memory_rule", "traffic_rule", "statement"):
        _require(type(memory.get(field)) is str, f"memory.{field} must be text")
    _require(
        "not a measured GPU footprint" in memory["statement"],
        "memory.statement must say that it is not a measured GPU footprint",
    )
    _require(
        set(memory)
        == {
            "unit",
            "memory_rule",
            "traffic_rule",
            "traffic_block_m",
            "measured",
            "statement",
            "rows",
        },
        "memory fields do not match",
    )


def _inputs(seed: int, family: str, index: int, shape: tuple[int, int, int]):
    generator = numpy.random.default_rng([seed, _STREAMS[family], index])
    q, k, v = (generator.standard_normal(shape, dtype=numpy.float32) for _ in "qkv")
    if family == "large":
        q, k = q * numpy.float32(_LARGE), k * numpy.float32(_LARGE)
    return q, k, v


def _digest(array: numpy.ndarray) -> str:
    return sha256(numpy.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def _reference(q, k, v, causal: bool) -> tuple[list, list]:
    """Float64 attention by ``P = exp(S - LSE)``, one head at a time."""

    n, d = q.shape[1], q.shape[2]
    scale = _scale(d)
    hidden = numpy.triu(numpy.ones((n, n), dtype=bool), 1) if causal else None
    outputs, lses = [], []
    for head in range(q.shape[0]):
        s = q[head].astype(numpy.float64) @ k[head].astype(numpy.float64).T * scale
        if hidden is not None:
            s[hidden] = -numpy.inf
        top = numpy.max(s, axis=1)
        lse = top + numpy.log(numpy.sum(numpy.exp(s - top[:, None]), axis=1))
        outputs.append(numpy.exp(s - lse[:, None]) @ v[head].astype(numpy.float64))
        lses.append(lse)
    return outputs, lses


def _bounds(q, k, v, causal: bool, block_n: int) -> list[tuple[float, float]]:
    n, d = q.shape[1], q.shape[2]
    scale = _scale(d)
    blocks = _ceil(n, block_n)
    gamma_d = d * _U / (1 - d * _U)
    terms = n + blocks
    rest = 2 * (terms * _U / (1 - terms * _U)) + 2 * _U * (blocks + 1)
    seen = numpy.tril(numpy.ones((n, n), dtype=bool)) if causal else slice(None)
    result = []
    for head in range(q.shape[0]):
        qa = numpy.abs(q[head].astype(numpy.float64))
        ka = numpy.abs(k[head].astype(numpy.float64))
        a_max = float(numpy.max((qa @ ka.T * scale)[seen]))
        s = q[head].astype(numpy.float64) @ k[head].astype(numpy.float64).T * scale
        s_max = float(numpy.max(numpy.abs(s)[seen]))
        sigma = gamma_d * a_max + 2 * _U * s_max
        log_error = 2 * sigma + 4 * _U * s_max + 8 * _U * (blocks + 1)
        v_max = float(numpy.max(numpy.abs(v[head])))
        result.append(
            (
                2.0 * (math.expm1(2 * log_error) + rest + _U) * v_max,
                2.0 * (log_error + rest),
            )
        )
    return result


def _decode(text: object, shape: tuple[int, ...], label: str) -> numpy.ndarray:
    _require(type(text) is str, f"{label} must be base64 text")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TritonFlashVerificationError(f"{label} is not base64") from error
    _require(
        len(raw) == math.prod(shape) * _FLOAT
        and base64.b64encode(raw) == text.encode(),
        f"{label} does not hold {math.prod(shape)} float32 values",
    )
    return numpy.frombuffer(raw, dtype="<f4").reshape(shape)


def _ratio(error: float, bound: float) -> float:
    if error == 0:
        return 0.0
    return error / bound if bound > 0 else math.inf


def _verify_run(
    run: object, *, plan: TritonFlashPlan, case, config, family: str, inputs, refs
) -> dict[str, float]:
    _require(
        isinstance(run, dict) and set(run) == _RUN_FIELDS,
        "a run record does not match the schema",
    )
    heads, n, d = case.batch_heads, case.sequence, case.head_dim
    bm, bn = config.block_m, config.block_n
    expected = {
        "config": f"m{bm}_n{bn}_w{config.num_warps}_s{config.num_stages}",
        "family": family,
        "programs": heads * _ceil(n, bm),
        "key_blocks": _ceil(n, bn),
        "guard_rows": _ceil(n, bm) * bm - n,
    }
    for field, value in expected.items():
        _compare(value, run[field], f"run.{field}")
    where = f"run {expected['config']} {family}"
    output = _decode(run["output"], (heads, n, d), f"{where} output")
    maxima = _decode(run["maxima"], (heads, n), f"{where} maxima")
    sums = _decode(run["sums"], (heads, n), f"{where} sums")
    for name, array in (("output", output), ("maxima", maxima), ("sums", sums)):
        _require(
            run[f"{name}_sha256"] == _digest(array),
            f"{where} {name}_sha256 does not match its values",
        )
        _require(bool(numpy.isfinite(array).all()), f"{where} {name} is not finite")
    _require(bool((sums >= 1).all()), f"{where} holds a running sum below 1")
    bounds = _bounds(*inputs, case.causal, bn)
    for field, column in (("output_bounds", 0), ("lse_bounds", 1)):
        _require(
            type(run[field]) is list and len(run[field]) == heads,
            f"{where} {field} must hold one bound per head",
        )
        for head, recorded in enumerate(run[field]):
            _close(recorded, bounds[head][column], f"{where} {field}[{head}]")
    outputs, lses = refs
    out_errors, lse_errors = [], []
    for head in range(heads):
        out_errors.append(float(numpy.max(numpy.abs(output[head] - outputs[head]))))
        rebuilt = maxima[head].astype(numpy.float64) + numpy.log(
            sums[head].astype(numpy.float64)
        )
        lse_errors.append(float(numpy.max(numpy.abs(rebuilt - lses[head]))))
    out_ratio = max(_ratio(e, b[0]) for e, b in zip(out_errors, bounds))
    lse_ratio = max(_ratio(e, b[1]) for e, b in zip(lse_errors, bounds))
    _require(out_ratio <= 1, f"{where} breaks the output bound: {out_ratio!r}")
    _require(lse_ratio <= 1, f"{where} breaks the log-sum-exp bound: {lse_ratio!r}")
    _close(run["max_output_error"], max(out_errors), f"{where} max_output_error")
    _close(run["output_bound_ratio"], out_ratio, f"{where} output_bound_ratio")
    _close(run["max_lse_error"], max(lse_errors), f"{where} max_lse_error")
    _close(run["lse_bound_ratio"], lse_ratio, f"{where} lse_bound_ratio")
    return {"output": run["output_bound_ratio"], "lse": run["lse_bound_ratio"]}


def _verify_case(plan: TritonFlashPlan, index: int, actual: object) -> list[dict]:
    case = plan.cases[index]
    _require(
        isinstance(actual, dict)
        and set(actual) == {"index", "case", "scale", "operands", "runs"},
        f"case {index} fields do not match",
    )
    _compare(index, actual["index"], f"case {index}.index")
    _compare(
        {
            "batch_heads": case.batch_heads,
            "sequence": case.sequence,
            "head_dim": case.head_dim,
            "causal": case.causal,
        },
        actual["case"],
        f"case {index}.case",
    )
    _compare(_scale(case.head_dim), actual["scale"], f"case {index}.scale")
    shape = (case.batch_heads, case.sequence, case.head_dim)
    prepared, operands = {}, []
    for family in _STREAMS:
        inputs = _inputs(plan.seed, family, index, shape)
        prepared[family] = (inputs, _reference(*inputs, case.causal))
        operands.append(
            {
                "family": family,
                **{
                    f"{name}_sha256": _digest(item) for name, item in zip("qkv", inputs)
                },
            }
        )
    _compare(operands, actual["operands"], f"case {index}.operands")
    runs = actual["runs"]
    _require(
        type(runs) is list and len(runs) == 2 * len(plan.configs),
        f"case {index} must hold two runs per config",
    )
    ratios = []
    position = 0
    for config in plan.configs:
        for family in _STREAMS:
            inputs, refs = prepared[family]
            ratios.append(
                _verify_run(
                    runs[position],
                    plan=plan,
                    case=case,
                    config=config,
                    family=family,
                    inputs=inputs,
                    refs=refs,
                )
            )
            position += 1
    return ratios


def _expected_summary(plan: TritonFlashPlan, ratios: list[dict]) -> dict[str, object]:
    steps = 2 * sum(
        case.batch_heads
        * _ceil(case.sequence, config.block_m)
        * _ceil(case.sequence, config.block_n)
        for config in plan.configs
        for case in plan.cases
    )
    stored = (
        2
        * len(plan.configs)
        * sum(c.batch_heads * c.sequence * (c.head_dim + 2) for c in plan.cases)
    )
    largest = max(
        (2 * n * n * _FLOAT + 4 * n * d * _FLOAT)
        / (2 * n * _FLOAT + 4 * n * d * _FLOAT)
        for n in _SEQUENCES
        for d in _MODEL_DIMS
    )
    return {
        "cases": len(plan.cases),
        "configs": len(plan.configs),
        "runs": len(ratios),
        "runs_within_bound": len(ratios),
        "causal_runs": 2 * len(plan.configs) * sum(case.causal for case in plan.cases),
        "max_output_bound_ratio": max(item["output"] for item in ratios),
        "max_lse_bound_ratio": max(item["lse"] for item in ratios),
        "interpreter_steps": steps,
        "stored_values": stored,
        "largest_modeled_memory_ratio": largest,
    }


def _environment(report: Mapping[str, object]) -> None:
    try:
        _matmul_environment(
            report["environment"], report["host_load"], report["recorded_at"]
        )
    except TritonMatmulVerificationError as error:
        raise TritonFlashVerificationError(str(error)) from error


def _verify(
    report: Mapping[str, object], plan: TritonFlashPlan | None
) -> TritonFlashPlan:
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
        embedded = TritonFlashPlan.from_mapping(report["plan"])
    except (ValueError, TypeError) as error:
        raise TritonFlashVerificationError(
            f"embedded plan is invalid: {error}"
        ) from error
    _compare(embedded.to_mapping(), report["plan"], "plan")
    if plan is not None:
        if type(plan) is not TritonFlashPlan:
            raise TypeError("plan must be TritonFlashPlan")
        _require(plan == embedded, "report was not built from this plan")
    _require(
        report["plan_sha256"] == _canonical_digest(report["plan"]),
        "plan_sha256 does not match the plan",
    )
    _environment(report)
    if report["label"] == INTERPRETER_LABEL:
        _require(
            report["environment"]["triton_interpret"] == "1",
            "an interpreter report must hold TRITON_INTERPRET=1",
        )
    _compare(_expected_kernel(embedded), report["kernel"], "kernel")
    _compare(
        {**rule_mapping(), "sentinel_float32_bits": _SENTINEL}, report["rule"], "rule"
    )
    _expected_memory(report["memory"])
    _compare(ci_install_mapping(), report["ci_install"], "ci_install")
    correctness = report["correctness"]
    _require(
        isinstance(correctness, dict)
        and set(correctness) == {"cases"}
        and type(correctness["cases"]) is list
        and len(correctness["cases"]) == len(embedded.cases),
        "correctness cases do not match the plan",
    )
    ratios = []
    for index, actual in enumerate(correctness["cases"]):
        ratios += _verify_case(embedded, index, actual)
    _compare(_expected_summary(embedded, ratios), report["summary"], "summary")
    return embedded


def verify_report(
    report: object, *, plan: TritonFlashPlan | None = None
) -> TritonFlashPlan:
    """Return the report's plan only if every field reproduces."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise TritonFlashVerificationError("report fields do not match the schema")
    try:
        return _verify(report, plan)
    except KernelLaunchVerificationError as error:
        raise TritonFlashVerificationError(str(error)) from error


def load_report(
    path: Path, *, plan: TritonFlashPlan | None = None
) -> dict[str, object]:
    """Load canonical report bytes and verify every field.

    A file that cannot be read raises ``TritonFlashError``. A file that was
    read but is not a canonical, reproducible report raises
    ``TritonFlashVerificationError``.
    """

    try:
        payload = read_bounded(path, limit=MAX_REPORT_BYTES, label="report")
    except KernelLaunchError as error:
        raise TritonFlashError(str(error)) from error
    try:
        report = loads_strict(payload, label="report")
    except KernelLaunchError as error:
        raise TritonFlashVerificationError(str(error)) from error
    _require(type(report) is dict, "report must be a JSON object")
    _require(
        payload == canonical_json_bytes(report) + b"\n",
        "report bytes are not canonical JSON",
    )
    verify_report(report, plan=plan)
    return report
