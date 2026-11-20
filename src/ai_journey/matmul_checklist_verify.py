"""Independent verification of sealed checklist reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
correctness fact from the seed and the stored digests. It uses the CPU
methods of the Day 76 verifier, not the code of this run: the float64
reference in k order, the fma chain replayed with a round-to-odd float64
step while the run repairs midpoints, the byte hashes, the launch geometry,
the nearest-rank statistics and the ratio rule. An output that records the
replay's bits must hash to the replay plus its sentinel guard tail and record
the replay's error. An output that records other bits cannot be rebuilt; it
must record another digest, a count of 1 to M * N differing elements and an
error within the bound.

Besides the label and rule texts of the run, the verifier uses three modules
of the run that measure nothing: the config parser checks the bounds, the
variant registry gives each kernel's text, group and tile, and the declared
checklist gives the hypotheses that the report must seal, byte for byte. It derives each pipeline verdict again
from the recorded limits, each launch and guard tail from the shape, the
model numbers from the tile sizes, and each statistic, rate, check, ratio,
verdict, held hypothesis, run total and summary from the raw samples. Every
section must match its rebuild as canonical JSON, so ``1``, ``1.0`` and
``true`` stay distinct. Timings cannot be observed again, so they are
checked for structure only, and torch is never imported.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement; any ``injected`` backend or timer verifies only as
synthetic. A matching digest shows internal consistency; it cannot prove
that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import re

import numpy

from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_checklist_benchmark import TRAFFIC_MODEL
from .matmul_checklist_config import MatmulChecklistConfig, config_from_mapping
from .matmul_checklist_report import (
    MEASUREMENT_BOUNDARY,
    REPLAY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
)
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_speedup import BEST_RULE, FRACTION_RULE, HYPOTHESIS_RULE
from .matmul_checklist_steps import (
    CHECKLIST,
    REFERENCE,
    checklist_mapping,
    checklist_sha256,
)
from .matmul_checklist_variants import VARIANTS, checklist_library
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    RELATIVE_ERROR,
)
from .matmul_metal_reference import REFERENCE as FLOAT64_REFERENCE
from .matmul_metal_report import FLOPS_PER_CALL, GFLOPS, TORCH_EQUIVALENT
from .matmul_metal_source import SHADER_SOURCE
from .matmul_tiled_benchmark import AGREEMENT
from .matmul_tiled_comparison import CUBLAS, RATIO_RULE
from .matmul_tiled_report import CORRECTNESS_RULES, SYNCHRONIZATION
from .matmul_tiled_verify import (
    MatmulTiledVerificationError,
    _calls,
    _checks,
    _digest,
    _environment,
    _exact,
    _gamma,
    _hash,
    _launch,
    _limits,
    _load,
    _output,
    _reference,
    _relative_error,
    _replay,
    _side,
    _step,
    _switches,
    _text_digest,
    _warnings,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_CORRECTNESS_STREAM = 0
_SENTINEL = 0x7FC00000
_U32 = 2.0**-24
_U64 = 2.0**-53
_TORCH = "torch_matmul"
_FEWER = "pipeline_allows_fewer_threads_than_the_tile"
_BYTES = "static_threadgroup_bytes_differ_from_the_tiles"
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "backend",
    "timer",
    "percentile_method",
    "config",
    "config_sha256",
    "checklist",
    "environment",
    "torch_mps_switches",
    "host_load",
    "warnings",
    "shaders",
    "refusals",
    "planned_work",
    "correctness",
    "benchmark",
    "speedup",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_CORRECTNESS_TEXT = {
    "reference": FLOAT64_REFERENCE,
    "replay": REPLAY,
    "relative_error": RELATIVE_ERROR,
    "bound": ERROR_BOUND,
    "gamma": GAMMA,
    "bound_source": BOUND_SOURCE,
    "sentinel_float32_bits": _SENTINEL,
    "rules": CORRECTNESS_RULES,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MatmulChecklistVerificationError(MatmulChecklistError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulChecklistVerificationError(message)


def _group(name: str, config: MatmulChecklistConfig) -> list[int]:
    group = VARIANTS[name].group_size
    return list(config.naive.group_size if group is None else group)


def _tile(name: str) -> tuple[int, int] | None:
    """The ``(BM, BN)`` tile of a kernel; ``None`` for a Day 74 kernel."""

    variant = VARIANTS[name]
    if variant.design is not None:
        return variant.design.block_m, variant.design.block_n
    if variant.group_size is not None:
        return variant.group_size[1], variant.group_size[0]
    return None


def _extent(name: str, m: int, n: int) -> list[int]:
    variant = VARIANTS[name]
    if variant.mapping == "x_rows":
        return [m, n]
    rows, columns = variant.outputs_per_thread
    return [-(-n // columns), -(-m // rows)]


def _guard(name: str, launch: dict[str, object], m: int, n: int) -> int:
    rows, columns = VARIANTS[name].outputs_per_thread
    x, y = launch["threads"]
    return x * y * rows * columns - m * n


def _admission(name: str, limits: dict[str, int]) -> dict[str, object]:
    x, y = VARIANTS[name].group_size
    threads, declared = x * y, VARIANTS[name].threadgroup_bytes
    reason = None
    if limits["max_threads_per_threadgroup"] < threads:
        reason = _FEWER
    elif limits["static_threadgroup_memory_bytes"] != declared:
        reason = _BYTES
    return {
        "name": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": threads,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "declared_threadgroup_bytes": declared,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
    }


def _shaders(section: object) -> tuple[dict, dict]:
    """Check the shader section; return every kernel's limits and each verdict."""

    entries = section.get("variants") if type(section) is dict else None
    _require(
        type(entries) is list and len(entries) == len(VARIANTS),
        "shaders must list every registered kernel",
    )
    limits, admissions, rebuilt = {}, {}, []
    for name, entry in zip(VARIANTS, entries):
        limits[name] = _limits(entry.get("limits") if type(entry) is dict else None)
        variant = VARIANTS[name]
        admission = None
        if variant.group_size is not None:
            admission = admissions[name] = _admission(name, limits[name])
        rebuilt.append(
            {
                **variant.to_mapping(),
                "source_sha256": _text_digest(variant.source),
                "limits": limits[name],
                "admission": admission,
            }
        )
    expected = {
        "naive_source_sha256": _text_digest(SHADER_SOURCE),
        "library_sha256": _text_digest(checklist_library()),
        "variants": rebuilt,
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return limits, admissions


def _cases(
    section: object, config: MatmulChecklistConfig, limits: dict, names: list[str]
) -> list[dict[str, object]]:
    shapes = config.correctness.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "correctness cases do not cover the config",
    )
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        generator = numpy.random.default_rng([config.seed, _CORRECTNESS_STREAM, index])
        left = generator.standard_normal((m, k), dtype=numpy.float32)
        right = generator.standard_normal((k, n), dtype=numpy.float32)
        total, scale = _reference(left, right)
        replay = _replay(left, right)
        bound = _gamma(k, _U32) + _gamma(k, _U64)
        error = _relative_error(replay, total, scale)
        _require(error <= bound, f"{label}: the replay exceeds the bound")
        launches = {
            name: _launch(_extent(name, m, n), _group(name, config), limits[name])
            for name in names
        }
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(names),
            f"{label} must record one output per kernel that ran",
        )
        for output, name in zip(outputs, names):
            guard = _guard(name, launches[name], m, n)
            _output(output, name, "k_order_fma", guard, (replay, error), m * n, bound)
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "launches": launches,
            "a_sha256": _hash(left),
            "b_sha256": _hash(right),
            "reference_sha256": _hash(total),
            "replay_sha256": _hash(replay),
            "bound": bound,
            "outputs": outputs,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
    return section


def _model(name: str, m: int, n: int, k: int) -> dict[str, float]:
    tile = _tile(name)
    if tile is None:
        loads = float(2 * k)
    else:
        rows, columns = tile
        loads = (-(-n // columns) * m * k + -(-m // rows) * k * n) / (m * n)
    return {"loads_per_output": loads, "flops_per_byte": 2 * k / (4 * (loads + 1))}


def _points(
    section: object, config: MatmulChecklistConfig, limits: dict, names: list[str]
) -> list[dict[str, object]]:
    shapes = config.benchmark.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "benchmark points do not cover the sweep",
    )
    sides = [*names, _TORCH]
    rebuilt = []
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        _require(type(entry) is dict, f"point {index} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        flops = 2 * m * n * k
        calls = _calls(flops, config.benchmark.target_flops_per_sample)
        recorded = entry.get("sides")
        _require(type(recorded) is dict, f"point {index} sides must be an object")
        agreement = 4 * _gamma(k, _U32)
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "flops_per_call": flops,
            "side_order": sides,
            "launches": {
                name: _launch(_extent(name, m, n), _group(name, config), limits[name])
                for name in names
            },
            "model": {name: _model(name, m, n, k) for name in names},
            "sides": {
                name: _side(recorded.get(name), name, calls, flops, config)
                for name in sides
            },
            "checks": _checks(entry.get("checks"), sides, agreement),
            "agreement_bound": agreement,
        }
        _require(
            _exact(entry, expected),
            f"point {index} at {[m, n, k]} does not recompute from its samples",
        )
        rebuilt.append(expected)
    return rebuilt


def _rate(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {
        "median_gflops_per_second": entry["median_gflops_per_second"],
        "iqr_gflops_per_second": entry["iqr_gflops_per_second"],
    }


def _speedup(points: list[dict], minimum: float, torch_version: str) -> dict:
    entries = []
    for point in points:
        sides = point["sides"]
        best = None
        for name in VARIANTS:
            if name in sides and (
                best is None
                or sides[name]["median_gflops_per_second"]
                > sides[best]["median_gflops_per_second"]
            ):
                best = name
        torch_rate = sides[_TORCH]["median_gflops_per_second"]
        rows = []
        for step in CHECKLIST:
            for variant in step.variants:
                entry, base = sides.get(variant), sides.get(step.base)
                to_base = (
                    None
                    if entry is None or base is None
                    else _step(entry, base, minimum)
                )
                predicted = list(step.predicted(variant))
                rows.append(
                    {
                        "step": step.step,
                        "item": step.item,
                        "variant": variant,
                        "base": step.base,
                        "ran": entry is not None,
                        "rate": _rate(entry),
                        "base_rate": _rate(base),
                        "to_base": to_base,
                        "to_s0": None
                        if entry is None
                        else _step(entry, sides[REFERENCE], minimum),
                        "fraction_of_torch": None
                        if entry is None
                        else entry["median_gflops_per_second"] / torch_rate,
                        "predicted": predicted,
                        "held": None
                        if to_base is None
                        else to_base["verdict"] in predicted,
                    }
                )
        entries.append(
            {
                "index": point["index"],
                "m": point["m"],
                "n": point["n"],
                "k": point["k"],
                "torch": _rate(sides[_TORCH]),
                "best": {
                    "variant": best,
                    "to_s0": _step(sides[best], sides[REFERENCE], minimum),
                    "fraction_of_torch": sides[best]["median_gflops_per_second"]
                    / torch_rate,
                },
                "rows": rows,
            }
        )
    hypotheses = []
    for step in CHECKLIST:
        for variant in step.variants:
            held = [
                (entry["index"], row["held"])
                for entry in entries
                for row in entry["rows"]
                if row["variant"] == variant
            ]
            hypotheses.append(
                {
                    "step": step.step,
                    "variant": variant,
                    "predicted": list(step.predicted(variant)),
                    "held_at": [index for index, value in held if value is True],
                    "missed_at": [index for index, value in held if value is False],
                    "no_verdict_at": [index for index, value in held if value is None],
                }
            )
    return {
        "reference": REFERENCE,
        "vendor": f"vendor GPU matmul: torch.matmul on MPS, PyTorch {torch_version}",
        "cublas": CUBLAS,
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": minimum,
        "points": entries,
        "hypotheses": hypotheses,
    }


def _planned(config: MatmulChecklistConfig) -> dict[str, int]:
    sweep = config.benchmark
    sides = len(VARIANTS) + 1
    samples = sides * (sweep.warmup + sweep.repeats)
    flops = [2 * shape.m * shape.n * shape.k for shape in sweep.shapes]
    batches = [_calls(work, sweep.target_flops_per_sample) for work in flops]
    cases = config.correctness.shapes
    return {
        "correctness_cases": len(cases),
        "replay_steps": sum(shape.m * shape.n * shape.k for shape in cases),
        "correctness_launches": len(cases) * len(VARIANTS),
        "benchmark_points": len(flops),
        "sides_per_point": sides,
        "timed_samples": sides * len(flops) * sweep.repeats,
        "calls": sum(samples * batch for batch in batches),
        "flops": sum(samples * batch * work for batch, work in zip(batches, flops)),
    }


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    gains: dict[str, object],
    refused: list[str],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    kernels = [output["kernel"] for output in cases[0]["outputs"]]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "refused_variants": refused,
        "cases_with_other_bits": {
            name: [
                case["index"]
                for case in cases
                if not case["outputs"][position]["replay_bits"]
            ]
            for position, name in enumerate(kernels)
        },
        "max_bound_ratio": max(
            output["bound_ratio"] for case in cases for output in case["outputs"]
        ),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "best": [
            {
                "m": entry["m"],
                "n": entry["n"],
                "k": entry["k"],
                "variant": entry["best"]["variant"],
                "over_s0": entry["best"]["to_s0"]["ratio"],
                "fraction_of_torch": entry["best"]["fraction_of_torch"],
                "torch_gflops_per_second": entry["torch"]["median_gflops_per_second"],
            }
            for entry in gains["points"]
        ],
        "hypotheses": {
            record["variant"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
            }
            for record in gains["hypotheses"]
        },
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _labels(report: dict[str, object]) -> tuple[bool, str, str]:
    _require(
        report["timer"] in (TIMER, INJECTED_TIMER)
        and report["backend"] in (BACKEND, INJECTED_BACKEND)
        and report["percentile_method"] == PERCENTILE_METHOD,
        "measurement method fields were changed",
    )
    measured = report["timer"] == TIMER and report["backend"] == BACKEND
    _require(
        report["measured"] is measured,
        "only the host timer and torch.mps.compile_shader make a measurement",
    )
    scope, label, boundary = (
        (MEASUREMENT_SCOPE, MEASUREMENT_LABEL, MEASUREMENT_BOUNDARY)
        if measured
        else (SYNTHETIC_SCOPE, SYNTHETIC_LABEL, SYNTHETIC_BOUNDARY)
    )
    _require(
        report["measurement_scope"] == scope
        and report["label"] == label
        and _exact(report["evidence_boundary"], boundary),
        "evidence labels or boundary were weakened",
    )
    return measured, scope, label


def _verify(
    report: object, supplied: MatmulChecklistConfig | None
) -> dict[str, object]:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    _require(
        type(report["schema_version"]) is int and report["schema_version"] == 1,
        "unsupported report schema",
    )
    _require(report["kind"] == REPORT_KIND, "unexpected report kind")
    _require(
        report[REPORT_DIGEST_FIELD] == _digest(report),
        "report digest does not match its content",
    )
    measured, scope, label = _labels(report)
    config = config_from_mapping(report["config"])
    _require(_exact(report["config"], config.to_mapping()), "config is not canonical")
    _require(
        isinstance(report["config_sha256"], str)
        and _SHA256.fullmatch(report["config_sha256"]) is not None
        and report["config_sha256"] == config.fingerprint(),
        "config digest does not match the embedded config",
    )
    if supplied is not None:
        _require(
            supplied.fingerprint() == config.fingerprint(),
            "report was not made from the supplied config",
        )
    _require(
        _exact(
            report["checklist"], {**checklist_mapping(), "sha256": checklist_sha256()}
        ),
        "the report does not seal the declared checklist and its hypotheses",
    )
    _environment(report["environment"], measured)
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    limits, admissions = _shaders(report["shaders"])
    refusals = [entry for entry in admissions.values() if not entry["admitted"]]
    _require(_exact(report["refusals"], refusals), "refusals do not match the limits")
    names = [
        name
        for name in VARIANTS
        if name not in admissions or admissions[name]["admitted"]
    ]
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, limits, names)
    _require(
        _exact(correctness, {**_CORRECTNESS_TEXT, "cases": cases}),
        "correctness section does not match the lab's reference and bound",
    )
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config, limits, names)
    expected = {
        "device": "mps",
        "flops_per_call": FLOPS_PER_CALL,
        "gflops": GFLOPS,
        "torch_equivalent": TORCH_EQUIVALENT,
        "synchronization": SYNCHRONIZATION,
        "agreement": AGREEMENT,
        "traffic_model": TRAFFIC_MODEL,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    torch_version = report["environment"]["torch_version"]
    gains = _speedup(points, config.comparison.min_faster_ratio, torch_version)
    _require(
        _exact(report["speedup"], gains),
        "the speedup rows or the hypotheses do not recompute from the samples",
    )
    timed = sum(
        sum(side["samples_ns"]) for point in points for side in point["sides"].values()
    )
    run = report["run"]
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "timed_ns"}
        and _exact(run["max_runtime_ns"], config.benchmark.max_runtime_seconds * _NS)
        and _exact(run["timed_ns"], timed)
        and type(run["elapsed_ns"]) is int
        and timed <= run["elapsed_ns"] <= run["max_runtime_ns"],
        "run time does not cover its samples within the budget",
    )
    summary = _summary(
        label,
        cases,
        points,
        gains,
        [entry["name"] for entry in refusals],
        captured,
        report["host_load"],
    )
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[REPORT_DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "checklist_sha256": report["checklist"]["sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "refused_variants": summary["refused_variants"],
        "cases_with_other_bits": summary["cases_with_other_bits"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "best": summary["best"],
        "hypotheses": summary["hypotheses"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object, *, config: MatmulChecklistConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 check, is reported as
    ``MatmulChecklistVerificationError``.
    """

    if config is not None and not isinstance(config, MatmulChecklistConfig):
        raise TypeError("config must be a MatmulChecklistConfig")
    try:
        return _verify(report, config)
    except MatmulChecklistVerificationError:
        raise
    except MatmulTiledVerificationError as error:
        raise MatmulChecklistVerificationError(str(error)) from error
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulChecklistVerificationError(
            f"report failed verification: {error}"
        ) from error
