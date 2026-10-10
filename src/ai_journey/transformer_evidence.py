"""Semantic validation and atomic publication of small CPU certification reports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import fields
from hashlib import sha256
from math import isfinite
from pathlib import Path

from .attention_audit import AttentionGradientAudit, AttentionWeightDiagnostics
from .transformer_certification import TransformerCertificationConfig
from .transformer_lab import TransformerConfig, TransformerLabError

SCHEMA = "ai-journey-transformer-certification-v1"


def _fingerprint(payload: dict) -> str:
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise TransformerLabError(f"invalid {name} fields")


def _number(value, name, *, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise TransformerLabError(f"invalid or failing {name}")


def verify_certification_report(report: dict) -> None:
    """Check the digest and measured safety conditions; no learner claim is implied."""
    _keys(report, {"schema", "result", "report_sha256"}, "report")
    if report["schema"] != SCHEMA:
        raise TransformerLabError("unsupported certification schema")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if not isinstance(report["report_sha256"], str) or report[
        "report_sha256"
    ] != _fingerprint(unsigned):
        raise TransformerLabError("certification fingerprint mismatch")
    result = report["result"]
    _keys(
        result,
        {
            "config",
            "input_sha256",
            "attention",
            "cache_rollover_error",
            "padding_error",
            "parameter_count",
            "runtime",
        },
        "result",
    )
    _keys(result["config"], {"model", "seed", "tolerance"}, "config")
    model = TransformerConfig(**result["config"]["model"])
    controls = TransformerCertificationConfig(
        model=model,
        seed=result["config"]["seed"],
        tolerance=result["config"]["tolerance"],
    )
    digest = result["input_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise TransformerLabError("invalid input fingerprint")
    attention = result["attention"]
    if not isinstance(attention, list) or len(attention) != model.layer_count:
        raise TransformerLabError("attention evidence must cover every layer")
    for layer in attention:
        _keys(layer, {"gradients", "causality", "probabilities"}, "attention")
        _keys(
            layer["gradients"],
            {field.name for field in fields(AttentionGradientAudit)},
            "gradients",
        )
        for name, value in layer["gradients"].items():
            _number(value, name, maximum=controls.tolerance)
        causal = layer["causality"]
        _keys(
            causal,
            {"prefix_error", "future_gradient", "boundaries_checked"},
            "causality",
        )
        for name in ("prefix_error", "future_gradient"):
            _number(causal[name], name, maximum=controls.tolerance)
        if (
            type(causal["boundaries_checked"]) is not int
            or causal["boundaries_checked"] != min(5, model.block_size) - 1
        ):
            raise TransformerLabError("incomplete causal boundaries")
        probabilities = layer["probabilities"]
        _keys(
            probabilities,
            {field.name for field in fields(AttentionWeightDiagnostics)},
            "probabilities",
        )
        for name in ("maximum_row_sum_error", "maximum_future_weight"):
            _number(probabilities[name], name, maximum=8 * 2.220446049250313e-16)
        _number(probabilities["minimum_probability"], "minimum_probability", maximum=1)
        for name in ("mean_entropy_by_head", "mean_normalized_entropy_by_head"):
            values = probabilities[name]
            if not isinstance(values, (list, tuple)) or len(values) != model.head_count:
                raise TransformerLabError("incomplete entropy diagnostics")
            for value in values:
                _number(
                    value, name, maximum=1 + 1e-12 if "normalized" in name else None
                )
    for name in ("cache_rollover_error", "padding_error"):
        _number(result[name], name, maximum=controls.tolerance)
    if type(result["parameter_count"]) is not int or result["parameter_count"] <= 0:
        raise TransformerLabError("invalid parameter count")
    runtime = result["runtime"]
    _keys(runtime, {"device", "dtype", "torch_version", "threads"}, "runtime")
    if (
        runtime["device"] != "cpu"
        or runtime["dtype"] != "float64"
        or not isinstance(runtime["torch_version"], str)
        or not runtime["torch_version"]
        or type(runtime["threads"]) is not int
        or runtime["threads"] <= 0
    ):
        raise TransformerLabError("invalid runtime scope")


def build_certification_report(result: dict) -> dict:
    # Canonical roundtrip owns nested data instead of aliasing the caller's dict.
    core = json.loads(json.dumps({"schema": SCHEMA, "result": result}, allow_nan=False))
    report = {**core, "report_sha256": _fingerprint(core)}
    verify_certification_report(report)
    return report


def write_certification_report(path: Path, report: dict) -> None:
    verify_certification_report(report)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_certification_report(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TransformerLabError("duplicate evidence field")
            result[key] = value
        return result

    if path.stat().st_size > 4_000_000:
        raise TransformerLabError("certification report exceeds size bound")
    report = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    verify_certification_report(report)
    return report
